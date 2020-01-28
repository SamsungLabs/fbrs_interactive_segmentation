import os
import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from tqdm import tqdm
import logging

from copy import deepcopy
from gluoncv.utils.viz.segmentation import DeNormalize
from collections import defaultdict

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint, get_dict_batchify_fn


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset, optimizer_params,
                 optimizer='adam',
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 start_epoch=0,
                 metrics=None,
                 additional_val_metrics=None,
                 hybridize_model=True,
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

        self.hybridize_model = hybridize_model
        self.checkpoint_interval = checkpoint_interval
        self.task_prefix = ''

        self.trainset = trainset
        self.valset = valset

        self.train_data = gluon.data.DataLoader(
            trainset, cfg.batch_size, shuffle=True,
            last_batch='discard',
            batchify_fn=get_dict_batchify_fn(cfg.workers),
            thread_pool=cfg.thread_pool,
            num_workers=cfg.workers)

        self.val_data = gluon.data.DataLoader(
            valset, cfg.val_batch_size,
            batchify_fn=get_dict_batchify_fn(cfg.workers),
            last_batch='discard',
            thread_pool=cfg.thread_pool,
            num_workers=cfg.workers)

        logger.info(model)
        model.cast('float32')
        model.collect_params().reset_ctx(ctx=cfg.ctx)

        self.net = model
        self.evaluator = None
        if cfg.weights is not None:
            if os.path.isfile(cfg.weights):
                model.load_parameters(cfg.weights, ctx=cfg.ctx, allow_missing=True)
                cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{cfg.weights}'")
        elif cfg.resume_exp is not None:
            self.load_checkpoint()

        if lr_scheduler is not None:
            optimizer_params['lr_scheduler'] = lr_scheduler(iters_per_epoch=len(self.train_data))
            optimizer_params['begin_num_update'] = start_epoch * len(self.train_data)

        kv = mx.kv.create(cfg.kvstore)

        self.trainer = gluon.Trainer(self.net.collect_params(),
                                     optimizer, optimizer_params,
                                     kvstore=kv, update_on_kvstore=len(cfg.ctx) > 1)

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        if cfg.input_normalization:
            self.denormalizator = DeNormalize(cfg.input_normalization['mean'],
                                              cfg.input_normalization['std'])
        else:
            self.denormalizator = lambda x: x

        self.sw = None
        self.image_dump_interval = image_dump_interval

    def training(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(logdir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        train_loss = 0.0
        hybridize = False

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            losses, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            autograd.backward(losses)
            self.trainer.step(1, ignore_stale_grad=True)

            batch_loss = sum(loss.asnumpy().mean() for loss in losses) / len(losses)
            train_loss += batch_loss

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=global_step)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=batch_loss, global_step=global_step)

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

            self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate', value=self.trainer.learning_rate,
                               global_step=global_step)

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}')
            for metric in self.train_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
            mx.nd.waitall()

            if self.hybridize_model and not hybridize:
                self.net.hybridize()
                hybridize = True

        for metric in self.train_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)

        save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix, epoch=None)
        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix, epoch=epoch)

    def validation(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(logdir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            losses, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            batch_loss = sum(loss.asnumpy()[0] for loss in losses) / len(losses)
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
        splitted_batch = {k: gluon.utils.split_and_load(v, ctx_list=self.cfg.ctx, even_split=False)
                               for k, v in batch_data.items()}
        if 'instances' in splitted_batch:
            splitted_batch['instances'] = [masks.reshape(shape=(-3, -2))
                                           for masks in splitted_batch['instances']]

        metrics = self.val_metrics if validation else self.train_metrics

        losses_logging = defaultdict(list)
        with autograd.record(True) if not validation else autograd.pause(False):
            outputs = [self.net(*net_inputs)
                       for net_inputs in zip(*(splitted_batch[input_name] for input_name in self.net_inputs))]

            losses = []
            for ictx, ctx_output in enumerate(outputs):
                loss = 0.0
                loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                     lambda: (ctx_output.instances, splitted_batch['instances'][ictx]))

                with autograd.pause():
                    for m in metrics:
                        m.update(*(getattr(ctx_output, x) for x in m.pred_outputs),
                                 *(splitted_batch[x][ictx] for x in m.gt_outputs))

                losses.append(loss)

        return losses, losses_logging, splitted_batch, outputs

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = mx.nd.mean(loss)
            losses_logging[loss_name].append(loss.asnumpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        outputs = outputs[0]

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

        image_blob, points = images[0][0], points[0][0]
        image = self.denormalizator(image_blob.as_in_context(mx.cpu(0))).asnumpy() * 255
        image = image.transpose((1, 2, 0))

        gt_instance_masks = instance_masks[0].asnumpy()
        predicted_instance_masks = mx.nd.sigmoid(outputs.instances).asnumpy()

        points = points.asnumpy()
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

    def load_checkpoint(self):
        cfg = self.cfg

        checkpoints = list(cfg.CHECKPOINTS_PATH.glob(f'{cfg.resume_prefix}*.params'))
        assert len(checkpoints) == 1

        checkpoint_path = checkpoints[0]
        logger.info(f'Load checkpoint from path: {checkpoint_path}')
        self.net.load_parameters(str(checkpoint_path), ctx=cfg.ctx, allow_missing=True)
