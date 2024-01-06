#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @project ->File  :DemoTorch->vgg_model
# @Time      :
# @Author    :
# @File      :vgg_model.py
# ===============【功能： 】================
import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # 展平与第一个全连接之间有Dropout
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, class_num))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        正向传播的过程
        :param x:
        :return:
        """
        # N X 3 X 224 X 224
        x = self.features(x)
        # N X 512 X 7 X 7  第0个维度是batch维度，所以从第1个维度开始
        x = torch.flatten(x, start_dim=1)
        # N X 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # self.modules()继承它的父类，nn.Module(), 它会返回一个迭代器，会遍历网络中所有的模块，
        # 会迭代我们定义的每一个层结构，
        for m in self.modules():  # 遍历每一层
            if isinstance(m, nn.Conv2d):
                # 对卷积权重w进行凯明初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    # 均值为0， 方差为0.01的正态分布
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias初始化为0
                    nn.init.constant_(m.bias, 0)



# 定义特征提取网络，make_feature
def make_feature(cfg:list):
    """

    :param cfg: 传入对应参数的配置列表即可
    :return:
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # 在vgg网络中，所有的参数都是kernel_size=2, stride=2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 通过非关键字形式的方式传入参数，
    return nn.Sequential(*layers)

cfgs = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# vgg11~vgg19分别代表A、B、D、E

#  实例化vgg网络
def vgg(model_name='vgg16', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Model number {} not in cfgs dict".format(model_name))

    model = VGG(make_feature(cfg), **kwargs)  # **kwargs可变长度的变量，包含classes_num和是否需要初始化
    return model


vgg_model = vgg(model_name="vgg16")




















