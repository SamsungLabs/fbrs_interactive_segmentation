import mxnet as mx
from mxnet import gluon

from .modeling.named_block import NamedHybridBlock
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead


def get_model(norm_layer, backbone_norm_layer=None, backbone='resnet50',
              deeplab_ch=256, aspp_dropout=0.5,
              use_rgb_conv=True, max_interactive_points=None):
    model = DistMapsModel(
        feature_extractor=DeepLabV3Plus(backbone=backbone,
                                        ch=deeplab_ch,
                                        project_dropout=aspp_dropout,
                                        norm_layer=norm_layer,
                                        backbone_norm_layer=backbone_norm_layer,
                                        backbone_lr_mult=0.1),
        head=SepConvHead(1, channels=deeplab_ch // 2, in_channels=deeplab_ch,
                         num_layers=2, norm_layer=norm_layer),
        max_interactive_points=max_interactive_points,
        use_rgb_conv=use_rgb_conv,
        norm_layer=norm_layer
    )

    return model


class DistMapsModel(NamedHybridBlock):
    def __init__(self, feature_extractor, head, norm_layer=gluon.nn.BatchNorm,
                 max_interactive_points=10, use_rgb_conv=True):
        super(DistMapsModel, self).__init__()

        with self.name_scope():
            if use_rgb_conv:
                self.rgb_conv = gluon.nn.HybridSequential()
                self.rgb_conv.add(
                    gluon.nn.Conv2D(channels=8, kernel_size=1),
                    gluon.nn.LeakyReLU(alpha=0.2),
                    norm_layer(),
                    gluon.nn.Conv2D(channels=3, kernel_size=1),
                )
            else:
                self.rgb_conv = None

            self.dist_maps = DistMaps(norm_radius=260, max_interactive_points=max_interactive_points,
                                      spatial_scale=1.0)
            self.feature_extractor = feature_extractor
            self.head = head

    def hybrid_forward(self, F, image, points):
        coord_features = self.dist_maps(image, F.reshape(points, shape=(-1, 2)))
        if self.rgb_conv is not None:
            x = self.rgb_conv(F.concat(image, coord_features, dim=1))
        else:
            c1, c2 = F.split(coord_features, num_outputs=2, axis=1)
            c3 = F.ones_like(c1)
            coord_features = F.concat(c1, c2, c3, dim=1)
            x = 0.8 * image * coord_features + 0.2 * image

        backbone_features = self.feature_extractor(x)

        instance_out = self.head(backbone_features[0])
        instance_out = F.contrib.BilinearResize2D(instance_out, image, mode='like')

        outputs = [('instances', instance_out)]
        return self.make_named_outputs(outputs)


class DistMaps(mx.gluon.HybridBlock):
    def __init__(self, norm_radius, max_interactive_points=0, spatial_scale=1.0):
        super(DistMaps, self).__init__()
        self.xs = None
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.max_interactive_points = max_interactive_points
        self._count = 0

    def _ctx_kwarg(self, x):
        if isinstance(x, mx.nd.NDArray):
            return {"ctx": x.context}
        return {}

    def get_coord_features(self, F, points, rows, cols, num_points,
                           **ctx_kwarg):
        row_array = F.arange(start=0, stop=rows, step=1, **ctx_kwarg)
        col_array = F.arange(start=0, stop=cols, step=1, **ctx_kwarg)

        coord_rows = F.broadcast_axis(F.reshape(row_array, (1, 1, rows, 1)), axis=(0, 3), size=(num_points, cols))
        coord_cols = F.broadcast_axis(F.reshape(col_array, (1, 1, 1, cols)), axis=(0, 2), size=(num_points, rows))

        coords = F.concat(coord_rows, coord_cols, dim=1)
        add_xy = F.reshape(points * self.spatial_scale, shape=(0, 0, 1, 1))
        add_xy = F.broadcast_axis(add_xy, axis=(2, 3), size=(rows, cols))

        coords = (coords - add_xy) / (self.norm_radius * self.spatial_scale)
        valid_points = F.min(points, axis=1, keepdims=False) >= 0
        exist_mask = F.reshape(valid_points, shape=(0, 1, 1, 1))
        coord_features = F.sqrt(F.sum(F.square(coords), axis=1, keepdims=1))
        coord_features = F.tanh(2 * coord_features)
        coord_features = F.where(F.broadcast_mul(exist_mask, coord_features + 1e-3) > 0,
                                 coord_features, F.ones_like(coord_features))

        coord_features = F.reshape(coord_features,
                                   shape=(-1, self.max_interactive_points, 1, rows, cols))
        coord_features = F.min(coord_features, axis=1, keepdims=False) # -> (bs * num_masks * 2) x 1 x h x w
        coord_features = F.reshape(coord_features, shape=(-1, 2, rows, cols))

        return coord_features

    def hybrid_forward(self, F, x, coords):
        if isinstance(x, mx.nd.NDArray):
            self.xs = x.shape
            self.coords_shape = coords.shape

        batch_size, rows, cols = self.xs[0], self.xs[2], self.xs[3]
        num_points = self.coords_shape[0]

        return self.get_coord_features(F, coords, rows, cols, num_points, **self._ctx_kwarg(x))
