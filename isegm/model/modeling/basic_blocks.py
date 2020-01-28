from mxnet import gluon


class ConvHead(gluon.HybridBlock):
    def __init__(self, num_outputs, channels=32, num_layers=1,
                 kernel_size=3, padding=1,
                 norm_layer=gluon.nn.BatchNorm):
        super(ConvHead, self).__init__()

        with self.name_scope():
            self.layers = gluon.nn.HybridSequential()
            for i in range(num_layers):
                self.layers.add(
                    gluon.nn.Conv2D(channels=channels, kernel_size=kernel_size,
                                    padding=padding, activation='relu'),
                    norm_layer(in_channels=channels)
                )
            self.layers.add(
                gluon.nn.Conv2D(channels=num_outputs, kernel_size=1, padding=0)
            )

    def hybrid_forward(self, F, *inputs):
        x = inputs[0]
        return self.layers(x)


class SepConvHead(gluon.HybridBlock):
    def __init__(self, num_outputs, channels, in_channels, num_layers=1,
                 kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0,
                 norm_layer=gluon.nn.BatchNorm):
        super(SepConvHead, self).__init__()

        with self.name_scope():
            self.layers = gluon.nn.HybridSequential()

            for i in range(num_layers):
                self.layers.add(
                    SeparableConv2D(channels,
                                    in_channels=in_channels if i == 0 else channels,
                                    dw_kernel=kernel_size, dw_padding=padding,
                                    norm_layer=norm_layer, activation='relu')
                )
                if dropout_ratio > 0 and dropout_indx == i:
                    self.layers.add(gluon.nn.Dropout(dropout_ratio))

            self.layers.add(
                gluon.nn.Conv2D(channels=num_outputs, kernel_size=1, padding=0)
            )

    def hybrid_forward(self, F, *inputs):
        x = inputs[0]

        return self.layers(x)


class SeparableConv2D(gluon.HybridBlock):
    def __init__(self, channels, in_channels, dw_kernel, dw_padding, dw_stride=1,
                 activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2D, self).__init__()
        self.body = gluon.nn.HybridSequential(prefix='')
        self.body.add(gluon.nn.Conv2D(in_channels, kernel_size=dw_kernel,
                                strides=dw_stride, padding=dw_padding,
                                use_bias=use_bias,
                                groups=in_channels))
        self.body.add(gluon.nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=use_bias))

        if norm_layer:
            self.body.add(norm_layer())
        if activation is not None:
            if isinstance(activation, str):
                self.body.add(gluon.nn.Activation(activation))
            else:
                self.body.add(activation())

    def hybrid_forward(self, F, x):
        return self.body(x)
