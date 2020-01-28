import random
from copy import deepcopy

import cv2
import numpy as np

from isegm.utils.misc import get_bbox_from_mask, expand_bbox, clamp_bbox
from albumentations import RandomCrop, CenterCrop


class ZoomIn(object):
    def __init__(self, augmentator, p=0.2, min_expand=1.25, max_expand=2.0):
        assert augmentator is not None
        self.augmentator, self.crop_size = self.get_zoom_in_augmentator(augmentator)
        self.p = p
        self.min_expand = min_expand
        self.max_expand = max_expand

    def __call__(self, sample):
        instances_info = {
            obj_id: obj_info for obj_id, obj_info in sample['instances_info'].items()
            if not obj_info['ignore']
        }

        obj_id, obj_info = random.choice(list(instances_info.items()))
        sample['instances_info'] = {obj_id: obj_info}
        obj_mask = sample['instances_mask'] == obj_id

        crop_height, crop_width = self.crop_size

        obj_bbox = get_bbox_from_mask(obj_mask)
        obj_bbox = fit_bbox_ratio(obj_bbox, crop_height / crop_width)

        expand_k = np.random.uniform(self.min_expand, self.max_expand)
        obj_bbox = expand_bbox(obj_bbox, expand_ratio=expand_k)
        obj_bbox = clamp_bbox(obj_bbox,
                              0, sample['image'].shape[0] - 1,
                              0, sample['image'].shape[1] - 1)

        sample['image'] = sample['image'][obj_bbox[0]:obj_bbox[1] + 1, obj_bbox[2]:obj_bbox[3]+1, :]
        sample['instances_mask'] = sample['instances_mask'][obj_bbox[0]:obj_bbox[1] + 1, obj_bbox[2]:obj_bbox[3]+1]

        sample['image'] = cv2.resize(sample['image'], (crop_width, crop_height))
        sample['instances_mask'] = cv2.resize(sample['instances_mask'], (crop_width, crop_height),
                                              interpolation=cv2.INTER_NEAREST)

        return sample

    @staticmethod
    def get_zoom_in_augmentator(augmentator):
        crop_augs = (RandomCrop, CenterCrop)
        zoom_in_augmentator = deepcopy(augmentator)
        zoom_in_augmentator.transforms = [
            x for x in zoom_in_augmentator.transforms
            if not isinstance(x, crop_augs)
        ]

        crop_height, crop_width = None, None
        for x in augmentator.transforms:
            if isinstance(x, crop_augs):
                crop_height, crop_width = x.height, x.width
                break

        assert crop_height is not None
        return zoom_in_augmentator, (crop_height, crop_width)


def fit_bbox_ratio(bbox, target_ratio):
    rmin, rmax, cmin, cmax = bbox

    bb_rc = 0.5 * (rmax + rmin)
    bb_cc = 0.5 * (cmax + cmin)
    bb_height = rmax - rmin + 1
    bb_width = cmax - cmin + 1
    bb_ratio = bb_height / bb_width

    if bb_ratio < target_ratio:
        bb_height = target_ratio * bb_width
    else:
        bb_width = bb_height / target_ratio

    rmin = bb_rc - 0.5 * bb_height
    rmax = bb_rc + 0.5 * bb_height
    cmin = bb_cc - 0.5 * bb_width
    cmax = bb_cc + 0.5 * bb_width

    return rmin, rmax, cmin, cmax
