import random

import cv2
import numpy as np


class BasePointSampler(object):
    def __init__(self, ignore_object_prob=0.0):
        self.ignore_object_prob = ignore_object_prob
        self._selected_mask = None
        self._selected_indices = None
        self._is_selected_ignore = False

    def sample_object(self, dataset_sample):
        raise NotImplementedError

    def sample_points(self):
        raise NotImplementedError

    @property
    def selected_mask(self):
        assert self._selected_mask is not None
        return self._selected_mask

    @selected_mask.setter
    def selected_mask(self, mask):
        if self._is_selected_ignore:
            mask = mask[mask > 0.5] = -1
        self._selected_mask = mask[np.newaxis, :].astype(np.float32)


class SinglePointSampler(BasePointSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_object(self, dataset_sample):
        if len(dataset_sample['objects_ids']) == 0:
            self._selected_mask = np.full(dataset_sample['instances_mask'].shape[:2], -1, dtype=np.float32)
            self._selected_indices = None
            self._is_selected_ignore = False
        else:
            if dataset_sample.get('ignore_ids', []) and random.random() < self.ignore_object_prob:
                random_id = random.choice(dataset_sample['ignore_ids'])
                mask = dataset_sample['ignore_mask'] == random_id
                self._is_selected_ignore = True
            else:
                random_id = random.choice(dataset_sample['objects_ids'])
                mask = dataset_sample['instances_mask'] == random_id
                self._is_selected_ignore = False

            self._selected_indices = np.argwhere(mask)
            self.selected_mask = mask

    def sample_points(self):
        assert self._selected_mask is not None
        if self._selected_indices is None:
            return [(-1, -1)]
        else:
            num_indices = self._selected_indices.shape[0]
            point = self._selected_indices[np.random.randint(0, num_indices)]
            return [point]


class MultiPointSampler(BasePointSampler):
    def __init__(self, max_num_points, prob_gamma=0.7, expand_ratio=0.1,
                 positive_erode_prob=0.9, positive_erode_iters=3,
                 negative_bg_prob=0.1, negative_other_prob=0.4, negative_border_prob=0.5,
                 merge_objects_prob=0.0, max_num_merged_objects=2,
                 **kwargs):
        kwargs['ignore_object_prob'] = 0.0
        super().__init__(**kwargs)
        self.max_num_points = max_num_points
        self.expand_ratio = expand_ratio
        self.positive_erode_prob = positive_erode_prob
        self.positive_erode_iters = positive_erode_iters
        self.merge_objects_prob = merge_objects_prob
        if max_num_merged_objects == -1:
            max_num_merged_objects = max_num_points
        self.max_num_merged_objects = max_num_merged_objects

        self.neg_strategies = ['bg', 'other', 'border']
        self.neg_strategies_prob = [negative_bg_prob, negative_other_prob, negative_border_prob]
        assert sum(self.neg_strategies_prob) == 1.0

        self._pos_probs = self._generate_probs(max_num_points, gamma=prob_gamma)
        self._neg_probs = self._generate_probs(max_num_points + 1, gamma=prob_gamma)
        self._neg_indices = None

    def sample_object(self, dataset_sample):
        if len(dataset_sample['objects_ids']) == 0:
            self.selected_mask = np.zeros_like(dataset_sample['instances_mask'])
            self._selected_indices = [[]]
            bg_indices = np.argwhere(dataset_sample['instances_mask'] == 0)
            self._neg_indices = {strategy: bg_indices for strategy in self.neg_strategies}
            return

        if len(dataset_sample['objects_ids']) > 1 and random.random() < self.merge_objects_prob:
            max_selected_objects = min(len(dataset_sample['objects_ids']), self.max_num_merged_objects)
            num_selected_objects = np.random.randint(2, max_selected_objects + 1)

            random_ids = random.sample(dataset_sample['objects_ids'], num_selected_objects)
            masks = [dataset_sample['instances_mask'] == obj_id for obj_id in random_ids]
            self._selected_indices = [np.argwhere(self._positive_erode(x)) for x in masks]
            mask = masks[0]
            for x in masks[1:]:
                mask = np.logical_or(mask, x)
        else:
            random_id = random.choice(dataset_sample['objects_ids'])
            mask = dataset_sample['instances_mask'] == random_id
            self._selected_indices = [np.argwhere(self._positive_erode(mask))]

        self.selected_mask = mask
        neg_indices_bg = np.argwhere(np.logical_not(mask))
        neg_indices_border = np.argwhere(self._get_border_mask(mask))
        if len(dataset_sample['objects_ids']) <= len(self._selected_indices):
            neg_indices_other = neg_indices_bg
        else:
            other_objects_mask = np.logical_and(dataset_sample['instances_mask'] > 0,
                                                np.logical_not(mask))
            neg_indices_other = np.argwhere(other_objects_mask)

        self._neg_indices = {
            'bg': neg_indices_bg,
            'other': neg_indices_other,
            'border': neg_indices_border
        }

    def sample_points(self):
        assert self._selected_mask is not None
        if len(self._selected_indices) == 1:
            pos_points = self._sample_points(self._selected_indices[0], is_negative=False)
        else:
            each_obj_points = [self._sample_points(indices, is_negative=False)
                               for indices in self._selected_indices]
            pos_points = [obj_points[0] for obj_points in each_obj_points]

            other_points_union = []
            for obj_points in each_obj_points:
                other_points_union.extend([t for t in obj_points[1:] if t[0] >= 0])

            num_additional_points = min(len(other_points_union), self.max_num_points - len(pos_points))
            if num_additional_points > 0:
                additional_points = random.sample(other_points_union, num_additional_points)
                assert num_additional_points + len(pos_points) <= self.max_num_points
                pos_points.extend(additional_points)

            random.shuffle(pos_points)
            if len(pos_points) < self.max_num_points:
                pos_points.extend([(-1, -1)] * (self.max_num_points - len(pos_points)))

        negative_strategy = np.random.choice(self.neg_strategies, p=self.neg_strategies_prob)
        neg_points = self._sample_points(self._neg_indices[negative_strategy], is_negative=True)

        return pos_points + neg_points

    def _sample_points(self, indices, is_negative=False):
        if is_negative:
            num_points = np.random.choice(np.arange(self.max_num_points + 1), p=self._neg_probs)
        else:
            num_points = 1 + np.random.choice(np.arange(self.max_num_points), p=self._pos_probs)

        points = []
        num_indices = len(indices)
        for j in range(self.max_num_points):
            point_coord = indices[np.random.randint(0, num_indices)] if j < num_points and num_indices > 0 else (-1, -1)
            points.append(point_coord)

        return points

    def _positive_erode(self, mask):
        if random.random() > self.positive_erode_prob:
            return mask

        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8),
                                kernel, iterations=self.positive_erode_iters).astype(np.bool)

        if eroded_mask.sum() > 10:
            return eroded_mask
        else:
            return mask

    def _get_border_mask(self, mask):
        expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask] = 0
        return expanded_mask

    @staticmethod
    def _generate_probs(max_num_points, gamma):
        probs = []
        last_value = 1
        for i in range(max_num_points):
            probs.append(last_value)
            last_value *= gamma

        probs = np.array(probs)
        probs /= probs.sum()

        return probs
