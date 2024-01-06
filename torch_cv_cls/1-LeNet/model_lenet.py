# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_lenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # (W-f + 2p)/s + 1 = (32 - 5 + 0)/1 + 1 = 28
        # 池化核为2, 步距大小也为2， 直接将原来的高度和宽度缩减为原来的一半。池化层只会改变它的高和宽
        # 不会影响它的深度，
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积，输入特征层的深度已经变成了16了, 采用32个卷积核， 卷积核的大小为5×5
        # (14 - 5 + 2 * 0)/1 + 1 = 10
        self.conv2 = nn.Conv2d(6, 16, 5)  # (32, 10, 10)
        self.maxpool2 = nn.MaxPool2d(2, 2)  #(32, 5, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x就是我们输入的数据， 对应pytorch中的通道排列顺序[batch, channel, height, width]
        x = F.relu(self.conv1(x))  # (16, 28, 28)
        x = self.maxpool1(x)  # (16, 14, 14)
        x = F.relu(self.conv2(x))  # (32, 10, 10)
        x = self.maxpool2(x)  # (32, 5, 5)
        # 全连接层的输入是一个一维的向量， 因此需要将我们的特征矩阵展平， 展平成一维向量，
        # 第一个全连接层的输入节点个数就是32 * 5 * 5，
        # 第一个维度是batch， -1表示让它自动推理，第二个维度也就是展平后的节点个数，
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        # 一般针对分类问题会将最后一层接上softmax层， 将我们的输出转化为概率分布， 理论上应该这么做，
        x = self.fc3(x)
        return x


# 接下来进行简单的测试
# [batch, channel, height, width]
# input1 = torch.rand([32, 3, 32, 32])
# # 实例化模型
# model = LeNet()
# print(model)
# out = model(input1)
# print(out)




