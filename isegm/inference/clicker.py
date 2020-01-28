from collections import namedtuple

import numpy as np
from scipy.ndimage import distance_transform_edt

Click = namedtuple('Click', ['is_positive', 'coords'])


class Clicker(object):
    def __init__(self, gt_mask, init_clicks=None, click_radius=1, ignore_label=-1):
        self.gt_mask = gt_mask == 1
        self.not_ignore_mask = gt_mask != ignore_label
        self.height, self.width = gt_mask.shape
        self.radius = click_radius

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self._add_click(click)

    def make_next_click(self, pred_mask):
        click = self._get_click(pred_mask)
        self._add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def get_clicks_maps(self):
        pos_clicks_map = self.pos_clicks_map.copy()
        neg_clicks_map = self.neg_clicks_map.copy()

        if self.radius > 0:
            pos_clicks_map = np.zeros_like(pos_clicks_map, dtype=np.bool)
            neg_clicks_map = pos_clicks_map.copy()

            for click in self.clicks_list:
                y, x = click.coords
                y1, x1 = y - self.radius, x - self.radius
                y2, x2 = y + self.radius + 1, x + self.radius + 1

                if click.is_positive:
                    pos_clicks_map[y1:y2, x1:x2] = True
                else:
                    neg_clicks_map[y1:y2, x1:x2] = True

        pos_clicks_map = pos_clicks_map.astype(np.float32)
        neg_clicks_map = neg_clicks_map.astype(np.float32)

        return pos_clicks_map, neg_clicks_map

    def _get_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = distance_transform_edt(fn_mask)
        fp_mask_dt = distance_transform_edt(fp_mask)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def _add_click(self, click):
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks += 1
            self.pos_clicks_map[coords[0], coords[1]] = True
        else:
            self.num_neg_clicks += 1
            self.neg_clicks_map[coords[0], coords[1]] = True

        self.not_clicked_map[coords[0], coords[1]] = False
        self.clicks_list.append(click)

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
            self.pos_clicks_map[coords[0], coords[1]] = False
        else:
            self.num_neg_clicks -= 1
            self.neg_clicks_map[coords[0], coords[1]] = False

        self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        self.pos_clicks_map = np.zeros_like(self.gt_mask, dtype=np.bool)
        self.neg_clicks_map = np.zeros_like(self.gt_mask, dtype=np.bool)
        self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)
