from mxnet import gluon
from collections import namedtuple


class NamedHybridBlock(gluon.HybridBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._outputs_type = None

    def make_named_outputs(self, outputs):
        keys = [x[0] for x in outputs]
        values = [x[1] for x in outputs]

        named_outputs = namedtuple('outputs', keys)(*values)
        self._outputs_type = type(named_outputs)
        return named_outputs

    def __call__(self, *args):
        out = super().__call__(*args)
        if not isinstance(out, self._outputs_type):
            out = self._outputs_type(*out)
        return out
