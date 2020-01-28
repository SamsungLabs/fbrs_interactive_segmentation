import math

import torch
import numpy as np

from isegm.inference.clicker import Click
from .base import BaseTransform


class Crops(BaseTransform):
    def __init__(self, crop_size=(320, 480), min_overlap=0.2):
        super().__init__()
        self.crop_height, self.crop_width = crop_size
        self.min_overlap = min_overlap

        self.x_offsets = None
        self.y_offsets = None
        self._counts = None
        self._image_crops = None

    def transform(self, image_nd, clicks_lists, clicks_maps=None):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        image_height, image_width = image_nd.shape[2:4]
        self._image_crops = None

        if image_height < self.crop_height or image_width < self.crop_width:
            return image_nd, clicks_lists, clicks_maps

        # if self._image_crops is None:
        self.x_offsets = get_offsets(image_width, self.crop_width, self.min_overlap)
        self.y_offsets = get_offsets(image_height, self.crop_height, self.min_overlap)
        self._counts = np.zeros((image_height, image_width))

        image_crops = []
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                self._counts[dy:dy + self.crop_height, dx:dx + self.crop_width] += 1
                image_crop = image_nd[:, :, dy:dy + self.crop_height, dx:dx + self.crop_width]
                image_crops.append(image_crop)
        self._image_crops = torch.cat(*image_crops, dim=0)
        self._counts = torch.tensor(self._counts, device=image_nd.device, dtype=torch.float32)

        clicks_list = clicks_lists[0]
        clicks_lists = []
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                crop_clicks = [Click(is_positive=x.is_positive, coords=(x.coords[0] - dy, x.coords[1] - dx))
                               for x in clicks_list]
                clicks_lists.append(crop_clicks)

        if clicks_maps is not None:
            pos_map, neg_map = clicks_maps
            pos_maps_crops = []
            neg_maps_crops = []
            for dy in self.y_offsets:
                for dx in self.x_offsets:
                    pos_maps_crops.append(pos_map[:, dy:dy + self.crop_height, dx:dx + self.crop_width])
                    neg_maps_crops.append(neg_map[:, dy:dy + self.crop_height, dx:dx + self.crop_width])
            clicks_maps = np.concatenate(pos_maps_crops, axis=0), np.concatenate(neg_maps_crops, axis=0)

        return self._image_crops, clicks_lists, clicks_maps

    def inv_transform(self, prob_map):
        if self._image_crops is None:
            return prob_map

        assert prob_map.shape[0] == self._image_crops.shape[0]
        new_prob_map = torch.zeros((1, 1, *self._counts.shape),
                                   dtype=prob_map.dtype, device=prob_map.device)

        crop_indx = 0
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                new_prob_map[0, 0, dy:dy + self.crop_height, dx:dx + self.crop_width] += prob_map[crop_indx, 0]
                crop_indx += 1
        new_prob_map = torch.div(new_prob_map, self._counts)

        return new_prob_map

    def reset(self):
        self.x_offsets = None
        self.y_offsets = None
        self._counts = None
        self._image_crops = None


def get_offsets(length, crop_size, min_overlap_ratio=0.2):
    if length == crop_size:
        return [0]

    N = (length / crop_size - min_overlap_ratio) / (1 - min_overlap_ratio)
    N = math.ceil(N)

    overlap_ratio = (N - length / crop_size) / (N - 1)
    overlap_width = int(crop_size * overlap_ratio)

    offsets = [0]
    for i in range(1, N):
        new_offset = offsets[-1] + crop_size - overlap_width
        if new_offset + crop_size > length:
            new_offset = length - crop_size

        offsets.append(new_offset)

    return offsets
