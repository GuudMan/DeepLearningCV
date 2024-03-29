# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_resnet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from tensorflow.keras import layers, Model, Sequential


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # ----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # ----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)
        return x


class Bottlenect(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottlenect, self).__init__()
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # ------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # ------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # ------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.add([identity, x])
        x = self.relu(x)
        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")
    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))
    return Sequential(layers_list, name=name)


def _resnet(block, block_num, im_width, im_height, num_classes=1000, include_top=True):
    # TensorFlow中tensor的通道顺序 NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME",
                      use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, block_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, block_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, block_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, block_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x
    model = Model(inputs=input_image, outputs=predict)
    return model


def resnet34(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_height, im_width, num_classes, include_top)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottlenect, [3, 4, 6, 3], im_height, im_width, num_classes, include_top)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottlenect, [3, 4, 23, 3], im_height, im_width, num_classes, include_top)


# import tensorflow as tf
# input = tf.random.uniform((8, 224, 224, 3))
# model = resnet34()
# print(model(input))