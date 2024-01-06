# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_googlenet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch.nn as nn
import torch
import torch.nn.functional as F
# pytorch官方参考代码：https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py

class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits
        # (224 - 7 + 2 * 3)/2 + 1 = 112 (3, 224, 224) -> (64, 112, 112)
        self.conv1 = BaseConv2d(in_channels=3, out_channels=64, kernel_size=7, padding=1, stride=2)
        # (64, 112, 112) -> (64, 56, 56) 看一下这里的参数是如何计算的，
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # (56 - 3 + 2 * 0)/1 + 1 = 56 (64, 56, 56) -> (64, 56, 56)
        self.conv2 = BaseConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # (56 - 3 + 2 * 0)/1 + 1 = 56 (64, 56, 56) -> (192, 56, 56)
        self.conv3 = BaseConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1)

        # (192, 56, 56) -> (192. 28. 28)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Inception3a 具体参数可查看表
        # in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj
        # ch1x1 + ch3x3 +  ch5x5 + pool_proj =output_channels
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)


        if self.aux_logits:
            # Inception4b  512
            self.aux1 = InceptionAux(512, num_classes)
            # Inception4e 528
            self.aux2 = InceptionAux(528, num_classes)
        # 指定输出固定尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        # [N, 3, 224, 224] -> [N, 64, 112, 112]
        x = self.conv1(x)
        # [N, 64, 112, 112] -> [N, 64, 56, 56]
        x = self.maxpool1(x)
        # [N, 64, 56, 56] -> [N, 56, 56, 64]
        x = self.conv2(x)
        # [N, 64, 56, 56] -> [N, 56, 56, 192]
        x = self.conv3(x)
        # [N, 56, 56, 192] -> [N, 28, 28, 192]
        x = self.maxpool2(x)

        # [N, 28, 28, 192] -> [N, 28, 28, 256]
        x = self.inception3a(x)
        # [N, 28, 28, 256] -> [N, 28, 28, 480]
        x = self.inception3b(x)
        # [N, 28, 28, 480] - > [N, 14, 14, 480]
        x = self.maxpool3(x)

        #  [N, 14, 14, 480] -> [N, 14, 14, 512]
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        #   [N, 14, 14, 512] -> [N, 14, 14, 512]
        x = self.inception4b(x)
        #   [N, 14, 14, 512] -> [N, 14, 14, 512]
        x = self.inception4c(x)
        #   [N, 14, 14, 512] -> [N, 14, 14, 528]
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        #   [N, 14, 14, 528] -> [N, 14, 14, 832]
        x = self.inception4e(x)
        # [N, 14, 14, 832] - > [N, 7, 7, 832]
        x = self.maxpool4(x)

        #  [N, 7, 7, 832] -> [N, 7, 7, 832]
        x = self.inception5a(x)
        #  [N, 7, 7, 832] -> [N, 7, 7, 1024]
        x = self.inception5b(x)

        #  [N, 7, 7, 1024] -> [N, 1, 1, 1024]
        x = self.avgpool(x)
        #  [N, 1, 1, 1024] -> [N, 1024]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        # [N, 1024] -> [N, num_classes]
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        """
        ch1x1: 
        ch3x3red: ch3x3reduce
        ch3x3:
        ch5x5red: ch5x5reduce
        """
        self.branch1 = BaseConv2d(in_channels, ch1x1, kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            BaseConv2d(in_channels=in_channels, out_channels=ch3x3red, kernel_size=1),
            # 保证输出大小等于输入大小
            BaseConv2d(in_channels=ch3x3red, out_channels=ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BaseConv2d(in_channels=in_channels, out_channels=ch5x5red, kernel_size=1),
            # 保证输出大小等于输入大小
            BaseConv2d(in_channels=ch5x5red, out_channels=ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # [batch, channel, h, w] torch.cat(outputs, 1)表示在channel维度上拼接
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagepool = nn.AvgPool2d(kernel_size=5, stride=3)
        # output [batch, 128, 4, 4]
        self.conv = BaseConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: [N, 512, 14, 14] aux2: [N, 528, 14, 14]
        x = self.averagepool(x)
        # aux1: [N, 512, 4, 4], aux2: [N, 528, 4, 4]
        x = self.conv(x)
        # [N, 128, 4, 4]
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # [N, 2048]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # [N, 2014]
        x = self.fc2(x)
        # [N, num_classes]
        return x


class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# input = torch.rand((16, 3, 224, 224))
# googlenet = GoogleNet(num_classes=5, aux_logits=False, init_weights=True)
# print(googlenet)
# output = googlenet(input)
# print(output)








































