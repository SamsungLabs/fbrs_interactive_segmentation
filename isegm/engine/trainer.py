import os
import logging
from copy import deepcopy
from collections import defaultdict

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 backbone_lr_mult=0.1,
                 net_inputs=('images', 'points')):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        self.train_data = DataLoader(
            trainset, cfg.batch_size, shuffle=True,
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size, shuffle=False,
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        backbone_params, other_params = model.get_trainable_params()
        opt_params = [
            {'params': backbone_params, 'lr': backbone_lr_mult * optimizer_params['lr']},
            {'params': other_params}
        ]
        if optimizer.lower() == 'adam':
            self.optim = torch.optim.Adam(opt_params, **optimizer_params)
        elif optimizer.lower() == 'adamw':
            self.optim = torch.optim.AdamW(opt_params, **optimizer_params)
        elif optimizer.lower() == 'sgd':
            self.optim = torch.optim.SGD(opt_params, **optimizer_params)
        else:
            raise NotImplementedError

        if cfg.multi_gpu:
            model = _CustomDP(model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0])

        logger.info(model)
        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        if cfg.input_normalization:
            mean = torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32)
            std = torch.tensor(cfg.input_normalization['std'], dtype=torch.float32)

            self.denormalizator = Normalize((-mean / std), (1.0 / std))
        else:
            self.denormalizator = lambda x: x

        self._load_weights()

    def training(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        train_loss = 0.0

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_loss = loss.item()
            train_loss += batch_loss

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                   value=np.array(loss_values).mean(),
                                   global_step=global_step)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall',
                               value=batch_loss,
                               global_step=global_step)

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

            self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                               value=self.lr if self.lr_scheduler is None else self.lr_scheduler.get_lr()[-1],
                               global_step=global_step)

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}')
            for metric in self.train_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        for metric in self.train_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                               value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)

        save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                        epoch=None, multi_gpu=self.cfg.multi_gpu)
        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            batch_loss = loss.item()
            val_loss += batch_loss
            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/num_batches:.6f}')
            for metric in self.val_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        for loss_name, loss_values in losses_logging.items():
            self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                               global_step=epoch, disable_avg=True)

        for metric in self.val_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)
        self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=val_loss / num_batches,
                           global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        if 'instances' in batch_data:
            batch_size, num_points, c, h, w = batch_data['instances'].size()
            batch_data['instances'] = batch_data['instances'].view(batch_size * num_points, c, h, w)
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = defaultdict(list)
        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, points = batch_data['images'], batch_data['points']

            output = self.net(image, points)

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], batch_data['instances']))
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: (output['instances_aux'], batch_data['instances']))
            with torch.no_grad():
                for m in metrics:
                    m.update(*(output.get(x) for x in m.pred_outputs),
                             *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name].append(loss.detach().cpu().numpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        image_blob, points = images[0], points[0]
        image = self.denormalizator(image_blob).cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()

        points = points.detach().cpu().numpy()
        if self.max_interactive_points > 0:
            points = points.reshape((-1, 2 * self.max_interactive_points, 2))
        else:
            points = points.reshape((-1, 1, 2))

        num_masks = points.shape[0]
        gt_masks = np.squeeze(gt_instance_masks[:num_masks], axis=1)
        predicted_masks = np.squeeze(predicted_instance_masks[:num_masks], axis=1)

        viz_image = []
        for gt_mask, point, predicted_mask in zip(gt_masks, points, predicted_masks):
            timage = draw_points(image, point[:max(1, self.max_interactive_points)], (0, 255, 0))
            if self.max_interactive_points > 0:
                timage = draw_points(timage, point[self.max_interactive_points:], (0, 0, 255))

            gt_mask[gt_mask < 0] = 0.25
            gt_mask = draw_probmap(gt_mask)
            predicted_mask = draw_probmap(predicted_mask)
            viz_image.append(np.hstack((timage, gt_mask, predicted_mask)))
        viz_image = np.vstack(viz_image)

        result = viz_image.astype(np.uint8)
        _save_image('instance_segmentation', result[:, :, ::-1])

    def _load_weights(self):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                self.net.load_weights(self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            self.net.load_weights(str(checkpoint_path))
        self.net = self.net.to(self.device)


class _CustomDP(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
