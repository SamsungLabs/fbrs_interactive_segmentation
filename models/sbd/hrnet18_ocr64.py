import random
from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import (
    Compose, ShiftScaleRotate, PadIfNeeded, RandomCrop,
    RGBShift, RandomBrightnessContrast, RandomRotate90, Flip
)

from isegm.engine.trainer import ISTrainer
from isegm.model.is_hrnet_model import get_hrnet_model
from isegm.model.losses import SigmoidBinaryCrossEntropyLoss, NormalizedFocalLossSigmoid
from isegm.model.metrics import AdaptiveIoU
from isegm.data.sbd import SBDDataset
from isegm.data.points_sampler import MultiPointSampler
from isegm.utils.log import logger
from isegm.model import initializer


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }
    model_cfg.num_max_points = 10

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'],
                             model_cfg.input_normalization['std']),
    ])

    model = get_hrnet_model(width=18, ocr_width=64, with_aux_output=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4

    num_epochs = 120
    num_masks = 1

    train_augmentator = Compose([
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    def scale_func(image_shape):
        return random.uniform(0.75, 1.25)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.7,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = SBDDataset(
        cfg.SBD_PATH,
        split='train',
        num_masks=num_masks,
        augmentator=train_augmentator,
        points_from_one_object=False,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        keep_background_prob=0.0,
        image_rescale=scale_func,
        points_sampler=points_sampler,
        samples_scores_path='./models/sbd/sbd_samples_weights.pkl',
        samples_scores_gamma=1.25
    )

    valset = SBDDataset(
        cfg.SBD_PATH,
        split='val',
        augmentator=val_augmentator,
        num_masks=num_masks,
        points_from_one_object=False,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        image_rescale=scale_func,
        points_sampler=points_sampler
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[100], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=5,
                        image_dump_interval=100,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points)
    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
