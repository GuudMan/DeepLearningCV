# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_googlenet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
from model_googlenet import GoogleNet
from torchvision import transforms


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = "./tulip.jpg"
    image = Image.open(image_path)

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = data_transform(image)

    # 增加维度
    image = torch.unsqueeze(image, dim=0)

    # read class_dict
    json_path = './4_GoogleNet/class_indices.json'
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    # create model
    googlenet = GoogleNet(num_classes=5, aux_logits=False, init_weights=True).to(device)
    # load model weight
    weights_path = "./4_GoogleNet/model/googlenet.pth"
    missing_keys, unexpected_keys = googlenet.\
        load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    googlenet.eval()


    with torch.no_grad():
        # predict class
        # [N, num_class]
        output = torch.squeeze(googlenet(image.to(device))).cpu()
        # [num_class]
        predict = torch.softmax(output, dim=0)
        print(predict)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {}, prob: {:.3f}".format(class_dict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10} prob: {:.3f}".format(class_dict[str(i)],
                                                 predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
