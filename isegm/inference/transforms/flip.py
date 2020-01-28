import torch
import numpy as np

from isegm.inference.clicker import Click
from .base import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def transform(self, image_nd, clicks_lists, clicks_maps=None):
        assert len(image_nd.shape) == 4
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [Click(is_positive=click.is_positive,
                                         coords=(click.coords[0], image_width - click.coords[1] - 1))
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        if clicks_maps is not None:
            pos_maps, neg_maps = clicks_maps
            clicks_maps = (np.concatenate((pos_maps, pos_maps[:, :, ::-1]), axis=0),
                           np.concatenate((neg_maps, neg_maps[:, :, ::-1]), axis=0))

        return image_nd, clicks_lists, clicks_maps

    def inv_transform(self, prob_map):
        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]

        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))

    def reset(self):
        pass
