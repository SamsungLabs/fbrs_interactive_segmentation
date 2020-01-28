import mxnet as mx


class BaseTransform(object):
    def __init__(self):
        self.image_changed = False

    def transform(self, image_nd, clicks_lists, clicks_maps=None):
        raise NotImplementedError

    def inv_transform(self, prob_map):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class SigmoidForPred(BaseTransform):
    def transform(self, image_nd, clicks_lists, clicks_maps=None):
        return image_nd, clicks_lists, clicks_maps

    def inv_transform(self, prob_map):
        return mx.nd.sigmoid(prob_map)

    def reset(self):
        pass
