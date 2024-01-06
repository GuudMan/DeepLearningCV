#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @project ->File  :DemoTensorflow->vgg_model
# @Time      :
# @Author    :
# @File      :vgg_model.py
# ===============【功能： 】================
from tensorflow.keras import layers, Sequential, models


def VGG(feature, im_height=224, im_width=224, class_num=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    # 原论文中为4096， 这里为减少计算量，选择为原论文的一半
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    output = layers.Softmax()(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model


def features(cfg):
    """
    通过配置文件搭建网络
    :param cfg:
    :return:
    """
    feature_layers = []
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            # tensorflow中只需指明输出维度， 不需要输入维度，也就是只需要卷积核的个数
            # vgg中所有卷积核的大小都是3， padding=1, stride也是1， 所以通过卷积之后，特征层的高和宽不变
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu")
            feature_layers.append(conv2d)

    # 给网络结构取了一个名字，叫feature
    return Sequential(feature_layers, name="feature")


cfgs = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 实例化模型
def vgg(model_name="vgg16", im_height=224, im_width=224, class_num=1000):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model name {} not in cfgs dict".format(model_name))
        exit(-1)
    model = VGG(features(cfg), im_height=im_height, im_width=im_width, class_num=class_num)
    return model

model = vgg(model_name='vgg16')