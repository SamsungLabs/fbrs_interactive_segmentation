import mxnet as mx
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .base import BasePredictor


class BRSBasePredictor(BasePredictor):
    def __init__(self, model, opt_functor, optimize_after_n_clicks=1, **kwargs):
        super().__init__(model, **kwargs)
        self.optimize_after_n_clicks = optimize_after_n_clicks
        self.opt_functor = opt_functor

        self.opt_data = None
        self.input_data = None

    def set_input_image(self, image_nd):
        super().set_input_image(image_nd)
        self.opt_data = None
        self.input_data = None

    def _get_clicks_maps(self, clicker):
        pos_map, neg_map = clicker.get_clicks_maps()
        return pos_map[np.newaxis, :], neg_map[np.newaxis, :]

    def _get_clicks_maps_nd(self, clicks_maps):
        pos_clicks_map, neg_clicks_map = clicks_maps
        with mx.autograd.pause(train_mode=False), mx.autograd.predict_mode():
            pos_clicks_map = mx.nd.array(pos_clicks_map, ctx=self.ctx)
            neg_clicks_map = mx.nd.array(neg_clicks_map, ctx=self.ctx)
            pos_clicks_map = mx.nd.expand_dims(pos_clicks_map, axis=1)
            neg_clicks_map = mx.nd.expand_dims(neg_clicks_map, axis=1)

        return pos_clicks_map, neg_clicks_map


class FeatureBRSPredictor(BRSBasePredictor):
    def __init__(self, model, opt_functor, insertion_mode='after_deeplab', **kwargs):
        super().__init__(model, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None

        if self.insertion_mode == 'after_deeplab':
            self.num_channels = model.feature_extractor.ch
        elif self.insertion_mode == 'after_c4':
            self.num_channels = model.feature_extractor.aspp_in_channels
        elif self.insertion_mode == 'after_aspp':
            self.num_channels = model.feature_extractor.ch + 32
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, clicks_maps, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_maps)
        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float32)

        if num_clicks <= self.net_clicks_limit or is_image_changed or self.input_data is None:
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = mx.nd.reshape(scale, (bs, -1, 1, 1))
            bias = mx.nd.reshape(bias, (bs, -1, 1, 1))
            if self.with_flip:
                scale = mx.nd.tile(scale, reps=(2, 1, 1, 1))
                bias = mx.nd.tile(bias, reps=(2, 1, 1, 1))

            scaled_backbone_features = mx.nd.broadcast_mul(self.input_data, scale)
            scaled_backbone_features = scaled_backbone_features + bias
            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = mx.nd.contrib.BilinearResize2D(x, self._c1_features, mode='like')
                x = mx.nd.concat(x, self._c1_features, dim=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)
            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = mx.nd.contrib.BilinearResize2D(pred_logits, image_nd, mode='like')
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, self.ctx)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                       **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]

        with mx.autograd.pause(train_mode=False), mx.autograd.predict_mode():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = mx.nd.array(self.opt_data, ctx=self.ctx)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with mx.autograd.pause(train_mode=False), mx.autograd.predict_mode():
            coord_features = self.net.dist_maps(image_nd, mx.nd.reshape(points, shape=(-1, 2)))
            x = self.net.rgb_conv(mx.nd.concat(image_nd, coord_features, dim=1))
            if self.insertion_mode == 'after_c4' or self.insertion_mode == 'after_aspp':
                c1, _, c3, c4 = self.net.feature_extractor.backbone(x)
                c1 = self.net.feature_extractor.skip_project(c1)

                if self.insertion_mode == 'after_aspp':
                    x = self.net.feature_extractor.aspp(c4)
                    x = mx.nd.contrib.BilinearResize2D(x, c1, mode='like')
                    x = mx.nd.concat(x, c1, dim=1)
                    backbone_features = x
                else:
                    backbone_features = c4
                    self._c1_features = c1
            else:
                backbone_features = self.net.feature_extractor(x)[0]

        return backbone_features


class InputBRSPredictor(BRSBasePredictor):
    def __init__(self, model, opt_functor, optimize_target='rgb', **kwargs):
        super().__init__(model, opt_functor=opt_functor, **kwargs)
        self.optimize_target = optimize_target

    def _get_prediction(self, image_nd, clicks_lists, clicks_maps, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_maps)
        num_clicks = len(clicks_lists[0])

        if self.opt_data is None or is_image_changed:
            opt_channels = 2 if self.optimize_target == 'dmaps' else 3
            bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]
            self.opt_data = mx.nd.zeros((bs, opt_channels, image_nd.shape[2], image_nd.shape[3]),
                                        ctx=self.ctx)

        def get_prediction_logits(opt_bias):
            input_image = image_nd
            if self.optimize_target == 'rgb':
                input_image = input_image + opt_bias
            dmaps = self.net.dist_maps(input_image, mx.nd.reshape(points_nd, shape=(-1, 2)))
            if self.optimize_target == 'dmaps':
                dmaps = dmaps + opt_bias

            x = self.net.rgb_conv(mx.nd.concat(input_image, dmaps, dim=1))
            if self.optimize_target == 'all':
                x = x + opt_bias

            backbone_features = self.net.feature_extractor(x)
            pred_logits = self.net.head(backbone_features[0])
            pred_logits = mx.nd.contrib.BilinearResize2D(pred_logits, image_nd, mode='like')

            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask, self.ctx,
                                    shape=self.opt_data.shape)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data.asnumpy().ravel(),
                                       **self.opt_functor.optimizer_params)

            self.opt_data = mx.nd.array(opt_result[0].reshape(self.opt_data.shape), ctx=self.ctx)

        with mx.autograd.pause(train_mode=False), mx.autograd.predict_mode():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_vars, _ = self.opt_functor.unpack_opt_params(self.opt_data)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits
