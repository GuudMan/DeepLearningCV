# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_googlenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import json
import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model_googlenet import GoogleNet


def main():
    im_height = 224
    im_width = 224
    img_path = "./tulip.jpg"
    img = Image.open(img_path)
    img = img.resize((im_height, im_width))

    img = ((np.array(img) / 255.0) - 0.5) / 0.5

    # add a batch
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = "./4_GoogleNet/class_indices.json"
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    model = GoogleNet(im_height=im_height, im_width=im_width, class_num=5, aux_logits=False)
    model.summary()
    weight_path = "./4_GoogleNet/model/googlenet.ckpt"
    model.load_weights(weight_path)

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    print_res = "class:{}, prob: {:.3}".format(class_dict[str(predict_class)], result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}, prob:{:.3}".format(class_dict[str(i)], result[i]))
    plt.show()


if __name__ == '__main__':
    main()
