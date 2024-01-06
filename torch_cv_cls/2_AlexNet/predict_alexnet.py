# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_alexnet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from model_alexnet import AlexNet
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch
import json


test_img = './9433167170_fa056d3175.jpg'
transform = transforms.Compose(
    [transforms.Resize((224, 224)), 
     transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

with open('./wz_class_indices.json', 'r+') as f:
    class_dict = json.load(f)
print(class_dict)


input = Image.open(test_img)
input = transform(input)  # [c, h, w]  torch.Size([3, 224, 224])
print(input.shape)
input = torch.unsqueeze(input, dim=0)  # [n, c, h, c]  torch.Size([1, 3, 224, 224])
print(input.shape)
alexnet = AlexNet()
alexnet.load_state_dict(torch.load("./wz_alexnet_best.pth"))
output = alexnet(input)  # [batch, 5]
flower_index = torch.max(output, dim=1)[1].item()
print(flower_index)
print(class_dict[str(flower_index)])








