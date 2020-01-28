import mxnet as mx
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.inference.utils import get_model_ctx_list


class BasePredictor(object):
    def __init__(self, net,
                 net_clicks_limit=None,
                 num_max_points=20,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.net = net
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.num_max_points = num_max_points
        self.original_image = None
        self.ctx = get_model_ctx_list(net)[0]
        self.zoom_in = zoom_in

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image_nd):
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.as_in_context(self.ctx)
        if len(self.original_image.shape) == 3:
            self.original_image = mx.nd.expand_dims(self.original_image, axis=0)

    def get_prediction(self, clicker):
        clicks_list = clicker.get_clicks()
        clicks_maps = self._get_clicks_maps(clicker)

        image_nd, clicks_lists, clicks_maps, is_image_changed = self.apply_transforms(
            self.original_image, [clicks_list], clicks_maps
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists, clicks_maps, is_image_changed)
        prediction = mx.nd.contrib.BilinearResize2D(pred_logits,
                                                    height=image_nd.shape[2],
                                                    width=image_nd.shape[3])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        return prediction.asnumpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, clicks_maps, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd).instances

    def apply_transforms(self, image_nd, clicks_lists, clicks_maps=None):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists, clicks_maps = t.transform(image_nd, clicks_lists, clicks_maps)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, clicks_maps, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (self.num_max_points - len(pos_clicks)) * [(-1, -1)]

            neg_clicks = [click.coords for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (self.num_max_points - len(neg_clicks)) * [(-1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return mx.nd.array(total_clicks, ctx=self.ctx)

    def _get_clicks_maps(self, clicker):
        return None
