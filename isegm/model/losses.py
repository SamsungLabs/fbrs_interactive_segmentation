import numpy as np
from mxnet import gluon
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like


class NormalizedFocalLossSigmoid(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1, **kwargs):
        super(NormalizedFocalLossSigmoid, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        one_hot = label > 0
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = F.sigmoid(pred)

        alpha = F.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = F.where(one_hot, pred, 1 - pred)
        pt = F.where(sample_weight, pt, F.ones_like(pt))
        beta = (1 - pt) ** self._gamma

        t_sum = F.sum(sample_weight, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)

        ignore_area = F.sum(label == self._ignore_label, axis=0, exclude=True).asnumpy()
        sample_mult = F.mean(mult, axis=0, exclude=True).asnumpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            bsum = F.sum(sample_weight, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(label == 1, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class SigmoidBinaryCrossEntropyLoss(Loss):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid
        assert ignore_label < 0

    def hybrid_forward(self, F, pred, label):
        label = _reshape_like(F, label, pred)
        sample_weight = label >= 0
        label = F.where(sample_weight, label, F.zeros_like(label))

        if not self._from_sigmoid:
            loss = F.relu(pred) - pred * label + \
                F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            eps = 1e-12
            loss = -(F.log(pred + eps) * label
                     + F.log(1. - pred + eps) * (1. - label))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
