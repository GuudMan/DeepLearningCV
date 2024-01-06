# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_lenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch
import torchvision.transforms as transforms
from PIL import Image

from model_lenet import LeNet
# def main():
# def main_function():
transforms = transforms.Compose(
    # 推理时， 下载的图片不可能那么标准
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],

)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
net.load_state_dict(torch.load('./LeNet.pth'))
im = Image.open('1.jpg')
im = transforms(im)  # [c, h, w]
# dim=0表示在最前面增加一个新的维度
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
# 不需要求损失梯度
with torch.no_grad():
    outputs = net(im)  # [batch, 10]
    print(outputs)
    # 计算输出中的最大值对应的index
    # print(torch.max(outputs, dim=1))
    # torch.max的输出为torch.return_types.max(values=tensor([1.6486]), indices=tensor([2]))
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])


























