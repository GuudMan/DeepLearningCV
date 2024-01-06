# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_googlenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from tensorflow.keras import layers, models, Model, Sequential
import tensorflow as tf


def InceptionV1(im_height=224, im_width=224, class_num=1000, aux_logits=False):
    # tensorflow通道顺序 NHWC
    # [None, 224, 224, 3]
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # [None, 224, 224, 3] -> [None, 112, 112, 64]
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME", use_bias=False,
                      name="conv1/conv")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/bn")(x)
    x = layers.ReLU()(x)

    # [None, 112, 112, 64] -> [None, 56, 56, 64]
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)

    #  [None, 56, 56, 64] ->  [None, 56, 56, 64]
    x = layers.Conv2D(filters=64, kernel_size=1, strides=1, use_bias=False, name="conv2/conv")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/bn")(x)
    x = layers.ReLU()(x)

    #  [None, 56, 56, 64] ->  [None, 56, 56, 192]  (56 - 3 + 2 * 1)/1 + 1 = 56
    x = layers.Conv2D(filters=192, kernel_size=3, strides=1, use_bias=False, padding="same", name="conv3/conv")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/bn")(x)
    x = layers.ReLU()(x)

    #  [None, 56, 56, 192] ->  [None, 28, 28, 192]
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    #  [None, 28, 28, 192] ->  [None, 28, 28, 256]
    x = Inception(64, 96, 128, 16, 32, 32, name="inception3a")(x)
    # [None, 28, 28, 256] -> [None, 28, 28, 480]
    x = Inception(128, 128, 192, 32, 96, 64, name="inception3b")(x)
    # [None, 28, 28, 480] -> [None, 14, 14, 480]
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    # [None, 14, 14, 480] -> [None, 14, 14, 512]
    x = Inception(192, 96, 208, 16, 48, 64, name="inception4a")(x)

    if aux_logits:
        aux1 = InceptionAux(class_num, name="aux1")(x)
    # [None, 14, 14, 512] -> [None, 14, 14, 512]
    x = Inception(160, 112, 224, 24, 64, 64, name="inception4b")(x)
    # [None, 14, 14, 512] -> [None, 14, 14, 512]
    x = Inception(128, 128, 256, 24, 64, 64, name="inception4c")(x)
    # [None, 14, 14, 512] -> [None, 14, 14, 528]
    x = Inception(112, 144, 288, 32, 64, 64, name="inception4d")(x)
    if aux_logits:
        aux2 = InceptionAux(class_num, name="aux2")(x)

    # [None, 14, 14, 528] -> [None, 14, 14, 832]
    x = Inception(256, 160, 320, 32, 128, 128, name="inception4e")(x)
    # [None, 14, 14, 832] -> [None, 7, 7, 832]
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    # [None, 7, 7, 832] -> [None, 7, 7, 832]
    x = Inception(256, 160, 320, 32, 128, 128, name="inception5a")(x)
    # [None, 7, 7, 832] -> [None, 7, 7, 1024]
    x = Inception(384, 192, 384, 48, 128, 128, name="inception5b")(x)
    # [None, 7, 7, 1024] -> [None, 1, 1, 1024]
    x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)
    # [None, 1, 1, 1024] -> [None, 1024*1*1]
    x = layers.Flatten(name="output_flatten")(x)
    #
    x = layers.Dropout(rate=0.4, name="output_dropout")(x)
    # [None, class_num]
    x = layers.Dense(class_num, name="output_dense")(x)

    aux3 = layers.Softmax(name="aux_3")(x)
    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__()
        self.branch1 = Sequential([
            layers.Conv2D(filters=ch1x1, kernel_size=1, use_bias=False, name="conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="bn"),
            layers.ReLU()], name="branch1")

        self.branch2 = Sequential([
            layers.Conv2D(filters=ch3x3red, kernel_size=1, use_bias=False, name="0/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="0/bn"),
            layers.ReLU(),

            layers.Conv2D(filters=ch3x3, kernel_size=3, padding="SAME", use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU()], name="branch2")

        self.branch3 = Sequential([
            layers.Conv2D(filters=ch5x5red, kernel_size=1, use_bias=False, name="0/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="0/bn"),
            layers.ReLU(),
            layers.Conv2D(filters=ch5x5, kernel_size=3, padding="SAME", use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU()], name="branch3")

        self.branch4 = Sequential([
            # caution: default stride=pool_size
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),
            layers.Conv2D(filters=pool_proj, kernel_size=1, use_bias=False, name="1/conv"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="1/bn"),
            layers.ReLU()], name="branch4")

    def call(self, input, **kwargs):
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs


class InceptionAux(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(InceptionAux, self).__init__()
        self.avgpool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, strides=1, use_bias=False, name="conv/conv")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv/bn")
        self.rule1 = layers.ReLU()

        self.fc1 = layers.Dense(units=1024, activation="relu", name="fc1")
        self.fc2 = layers.Dense(units=num_classes, name="fc2")
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        # aux1 [None, 14, 14, 512] aux2 [None, 14, 14, 528]
        # aux1: [None, 14, 14, 512] -> [None, 4, 4, 512]  (14 - 5)/3 + 1 = 4
        # axu2: [None, 14, 14, 528] -> [None, 4, 4, 528]  (14 - 5)/3 + 1 = 4
        x = self.avgpool(inputs)
        # aux1 [None, 4, 4, 512]-> [4, 4, 512]  aux2 [None, 4, 4, 528]-> [4, 4, 528]
        x = self.conv(x)
        #
        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.softmax(x)
        return x


# input = tf.random.uniform((16, 224, 224, 3))
# googlenet = GoogleNet(class_num=5, aux_logits=False)
# output = googlenet(input)
# print(output)
