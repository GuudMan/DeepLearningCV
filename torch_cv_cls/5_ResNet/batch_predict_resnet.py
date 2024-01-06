# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : batch_predict_resnet.py
# @Time       ：2023/10/25 17:08
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model_resnet import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 加载需要遍历预测的文件夹
    img_root = "./data/flower_data/val/roses"
    # print(os.listdir(img_root))
    img_path_list = [os.path.join(img_root, i) for
                     i in os.listdir(img_root) if i.endswith(".jpg")]
    # read class_indict
    json_path = "./5_ResNet/class_indices.json"

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classe=5).to(device)

    # load weight
    weights_path = "./5_ResNet/model/resnet34.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # 每次预测时， 最多将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file:{img_path} does not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("imags:{} class:{} prob:{:.3}".format(img_path_list[ids * batch_size + idx],
                                                            class_indict[str(cla.numpy())],
                                                            pro.numpy()))
if __name__ == '__main__':
    main()


