# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_mobilenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from tensorflow.keras import layers, Model, Sequential


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    the function is taken from the origianl tf repo
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                                  padding="SAME", use_bias=False, name="Conv2d")
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1E-5, name="BatchNorm")
        self.activation = layers.ReLU(max_value=6.0)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(layers.Layer):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__()
        self.hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer_list = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layer_list.append(ConvBNReLU(out_channel=self.hidden_channel, kernel_size=1, name="expand"))
        layer_list.extend([
            # 3x3 depthwise conv
            layers.DepthwiseConv2D(kernel_size=3, padding="SAME", strides=stride,
                                   use_bias=False, name="depthwise"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="depthwise/BatchNorm"),
            layers.ReLU(max_value=6.0),
            # 1x1 pointwise conv
            layers.Conv2D(filters=out_channel, kernel_size=1, strides=1,
                          padding="SAME", use_bias=False, name="project"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="project/BatchNorm")
        ])
        self.main_branch = Sequential(layer_list, name="expanded_conv")

    def call(self, inputs, training=False, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs, training=training)

def MobileNetV2(im_height=224,
                im_width=224,
                num_classes=1000,
                alpha=1.0,
                round_nearest=8,
                include_top=True):
    block = InvertedResidual
    input_channel = _make_divisible(32 * alpha, round_nearest)
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    inverted_residual_setting = [
        # t: 扩展因子， c: 输出特征矩阵深度channel, n: bottleneck的重复次数 s: 步距
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # conv1
    x = ConvBNReLU(input_channel, stride=2, name="Conv")(input_image)
    # building inverted residual blocks
    for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            x = block(x.shape[-1],
                      output_channel,
                      stride,
                      expand_ratio=t)(x)
    # building last several layers
    x = ConvBNReLU(last_channel, kernel_size=1, name="Conv_1")(x)

    if include_top is True:
        # bulid classifier
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, name="Logits")(x)
    else:
        output = x
    model = Model(inputs=input_image, outputs=output)
    return model

# import tensorflow as tf
# input = tf.random.uniform(shape=(8, 224, 224, 3))
# mobilenetv2 = MobileNetV2(num_classes=5)
# output = mobilenetv2(input)
# print(output)

