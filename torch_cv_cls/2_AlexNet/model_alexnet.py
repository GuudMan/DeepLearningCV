# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_alexnet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, kernel_size=11, out_channels=48, padding=2, stride=4)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=48, kernel_size=5, out_channels=128, padding=2, stride=1)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=192, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=192, kernel_size=3, out_channels=192, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=192, kernel_size=3, out_channels=128, padding=1, stride=1)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpooling3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


# input = torch.rand([32, 3, 224, 224])
# alexnet = AlexNet()
# print(alexnet)
# output = alexnet(input)
# print(output)
