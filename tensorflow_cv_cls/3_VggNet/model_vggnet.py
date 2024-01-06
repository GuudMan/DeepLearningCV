# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_vggnet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道顺序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(num_classes, activation='relu',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    output = layers.Softmax()(x)

    model = Model(inputs=input_image, outputs=output)
    return model


def make_feature(cfg):
    feature_layers = []
    for v in cfg:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            feature_layers.append(layers.Conv2D(v, kernel_size=3, padding="same", activation="relu",
                                                kernel_initializer=CONV_KERNEL_INITIALIZER))
    return Sequential(feature_layers, name="feature")


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=1000):
    cfg = cfgs[model_name]
    model = VGG(make_feature(cfg), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model


# input = tf.random.uniform((4, 224, 224, 3))
# vggnet = vgg("vgg16", num_classes=5)
# print(vggnet.summary())
# print(vggnet(input))


