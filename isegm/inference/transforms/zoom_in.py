import cv2
import torch
import numpy as np

from isegm.inference.clicker import Click
from isegm.utils.misc import get_bbox_iou, get_bbox_from_mask, expand_bbox, clamp_bbox
from .base import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(self,
                 target_size=400,
                 skip_clicks=1,
                 expansion_ratio=1.4,
                 min_crop_size=200,
                 recompute_thresh_iou=0.5,
                 prob_thresh=0.50):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None

    def transform(self, image_nd, clicks_lists, clicks_maps=None):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists, clicks_maps

        self._input_image = image_nd

        current_object_roi = None
        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh).detach().cpu().numpy()[0, 0]
            if current_pred_mask.sum() > 0:
                current_object_roi = get_object_roi(current_pred_mask, clicks_list,
                                                    self.expansion_ratio, self.min_crop_size)

        if current_object_roi is None:
            return image_nd, clicks_lists, clicks_maps

        update_object_roi = False
        if self._object_roi is None:
            update_object_roi = True
        elif not check_object_roi(self._object_roi, clicks_list):
            update_object_roi = True
        elif get_bbox_iou(current_object_roi, self._object_roi) < self.recompute_thresh_iou:
            update_object_roi = True

        if update_object_roi:
            self._object_roi = current_object_roi
            self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)
            self.image_changed = True

        tclicks_lists = [self._transform_clicks(clicks_list)]
        tclicks_maps = self._transform_click_maps(clicks_maps)
        return self._roi_image, tclicks_lists, tclicks_maps

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(prob_map, size=(rmax - rmin + 1, cmax - cmin + 1),
                                                   mode='bilinear', align_corners=True)

        new_prob_map = torch.zeros_like(self._prev_probs) if self._prev_probs is not None else prob_map
        new_prob_map[:, :, rmin:rmax+1, cmin:cmax+1] = prob_map
        self._prev_probs = new_prob_map

        return new_prob_map

    def check_possible_recalculation(self):
        if self._prev_probs is None or self._object_roi is not None or self.skip_clicks > 0:
            return False

        pred_mask = (self._prev_probs > self.prob_thresh).asnumpy()[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(pred_mask, [],
                                                 self.expansion_ratio, self.min_crop_size)
            image_roi = (0, self._input_image[2] - 1, 0, self._input_image[3] - 1)
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def reset(self):
        self._input_image = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(Click(is_positive=click.is_positive, coords=(new_r, new_c)))
        return transformed_clicks

    def _transform_click_maps(self, clicks_maps):
        if self._object_roi is None or clicks_maps is None:
            return clicks_maps

        assert clicks_maps[0].shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]
        pos_maps, neg_maps = clicks_maps
        pos_maps = pos_maps[:, rmin:rmax+1, cmin:cmax+1]
        neg_maps = neg_maps[:, rmin:rmax + 1, cmin:cmax + 1]

        if max(pos_maps.shape[1], pos_maps.shape[2]) > max(crop_width, crop_height):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        pos_maps = cv2.resize(pos_maps[0], dsize=(crop_width, crop_height),
                              interpolation=interpolation)[np.newaxis, :]
        neg_maps = cv2.resize(neg_maps[0], dsize=(crop_width, crop_height),
                              interpolation=interpolation)[np.newaxis, :]
        pos_maps = pos_maps / (pos_maps.max() + 1e-4)
        neg_maps = neg_maps / (neg_maps.max() + 1e-4)

        return pos_maps, neg_maps


def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]
        roi_image_nd = torch.nn.functional.interpolate(roi_image_nd, size=(new_height, new_width),
                                                       mode='bilinear', align_corners=True)

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True
