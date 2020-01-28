import mxnet as mx
from mxnet import gluon
from isegm.model.losses import SigmoidBinaryCrossEntropyLoss


class BRSMaskLoss(gluon.Block):
    def __init__(self, eps=1e-5):
        super().__init__()
        self._eps = eps

    def forward(self, result, pos_mask, neg_mask):
        pos_diff = mx.nd.broadcast_mul(1 - result, pos_mask)
        pos_target = mx.nd.sum(pos_diff ** 2)
        pos_target = mx.nd.broadcast_div(pos_target, mx.nd.sum(pos_mask) + self._eps)

        neg_diff = mx.nd.broadcast_mul(result, neg_mask)
        neg_target = mx.nd.sum(neg_diff ** 2)
        neg_target = mx.nd.broadcast_div(neg_target, mx.nd.sum(neg_mask) + self._eps)
        
        loss = pos_target + neg_target

        with mx.autograd.pause(train_mode=False):
            f_max_pos = mx.nd.max(mx.nd.abs(pos_diff)).asscalar()
            f_max_neg = mx.nd.max(mx.nd.abs(neg_diff)).asscalar()

        return loss, f_max_pos, f_max_neg


class OracleMaskLoss(mx.gluon.Block):
    def __init__(self):
        super().__init__()
        self.gt_mask = None
        self.loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self.predictor = None
        self.history = []

    def set_gt_mask(self, gt_mask):
        self.gt_mask = gt_mask
        self.history = []

    def forward(self, result, pos_mask, neg_mask):
        gt_mask = self.gt_mask.as_in_context(result.context)
        if self.predictor.object_roi is not None:
            r1, r2, c1, c2 = self.predictor.object_roi[:4]
            gt_mask = gt_mask[:, :, r1:r2 + 1, c1:c2 + 1]
            gt_mask = mx.nd.contrib.BilinearResize2D(gt_mask, result, mode='like')

        if result.shape[0] == 2:
            gt_mask_flipped = mx.nd.flip(gt_mask, axis=3)
            gt_mask = mx.nd.concat(gt_mask, gt_mask_flipped, dim=0)

        loss = self.loss(result, gt_mask)
        self.history.append(loss.asnumpy()[0])

        if len(self.history) > 5 and abs(self.history[-5] - self.history[-1]) < 1e-5:
            return 0, 0, 0

        return loss, 1.0, 1.0
