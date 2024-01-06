#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @project ->File  :DemoTorch->alenet_model
# @Time      :
# @Author    :
# @File      :alexnet_model.py
# ===============【功能： 】================
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # nn.Sequential将一系列层结构进行打包， 对于网络层次比较多的网络，使用这种方式可以减少工作量
        self.features = nn.Sequential(
            # 原论文中卷积核的个数是96， 这里为减少计算设置为48 (w-f+2p)/s+1=
            # input (3, 224, 224) -> (224 - 11 + 2* 2)/4 + 1=55.25  (48, 55, 55)
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # inplace 通过这个方法在内存中载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output (48, 27, 27)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output (128, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output (128 13, 13)
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output (192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output (192 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output (128, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   # output (128 6, 6)  128*6*6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 默认p=0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            # 初始化权重函数
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 第0维度的batch， 所有从1开始，也就是从channel维度开始展平， 也可用view函数
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        # self.modules()继承它的父类，nn.Module(), 它会返回一个迭代器，会遍历网络中所有的模块，
        # 会迭代我们定义的每一个层结构，
        for m in self.modules():
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

