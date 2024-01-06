# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_resnet.py
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
from model_resnet import resnet34
from torchvision import transforms


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = "./tulip.jpg"
    image = Image.open(image_path)

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = data_transform(image)

    # 增加维度
    image = torch.unsqueeze(image, dim=0)

    # read class_dict
    json_path = './5_ResNet/class_indices.json'
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    # create model
    resnet = resnet34(num_classe=5).to(device)
    # load model weight
    weights_path = "./5_ResNet/model/resnet34.pth"
    resnet.load_state_dict(torch.load(weights_path, map_location=device))
    
    # prediction
    resnet.eval()
    with torch.no_grad():
        # predict class
        # [N, num_class]
        output = torch.squeeze(resnet(image.to(device))).cpu()
        print(output)
        # [num_class]
        predict = torch.softmax(output, dim=0)
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