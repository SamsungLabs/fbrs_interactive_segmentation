import torch
import torch.nn as nn
from torch import nn as nn

import isegm.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class DistMaps(nn.Module):
    def __init__(self, norm_radius, max_interactive_points=0, spatial_scale=1.0):
        super(DistMaps, self).__init__()
        self.xs = None
        self.coords_shape = None
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.max_interactive_points = max_interactive_points
        self._count = 0

    def get_coord_features(self, points, rows, cols, num_points):
        row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float, device=points.device)
        col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float, device=points.device)

        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
        coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(num_points, 1, 1, 1)
        add_xy = (points * self.spatial_scale).view(num_points, points.size(1), 1, 1)

        coords = (coords - add_xy) / (self.norm_radius * self.spatial_scale)
        valid_points = torch.min(points, dim=1, keepdim=False)[0] >= 0
        exist_mask = valid_points.view(num_points, 1, 1, 1)
        coord_features = torch.sqrt(torch.sum(coords**2, dim=1, keepdim=True))
        coord_features = torch.tanh(2 * coord_features)
        coord_features = torch.where(exist_mask * (coord_features + 1e-3) > 0,
                                     coord_features, torch.ones_like(coord_features))

        coord_features = coord_features.view(-1, self.max_interactive_points, 1, rows, cols)
        coord_features = torch.min(coord_features, dim=1, keepdim=False)[0]  # -> (bs * num_masks * 2) x 1 x h x w
        coord_features = coord_features.view(-1, 2, rows, cols)

        return coord_features

    def forward(self, x, coords):
        self.xs = x.shape
        self.coords_shape = coords.shape

        batch_size, rows, cols = self.xs[0], self.xs[2], self.xs[3]
        num_points = self.coords_shape[0]

        return self.get_coord_features(coords, rows, cols, num_points)