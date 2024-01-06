# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_resnet.py
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
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model_resnet import resnet50


def main():
    im_height = 224
    im_width = 224
    num_classes = 5

    img_path = "./tulip.jpg"
    img = Image.open(img_path)
    img = img.resize((im_height, im_width))

    # scaling pixel value to (0-1)
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    img = np.array(img).astype(np.float32)
    img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

    # add a batch
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = "./5_ResNet/class_indices.json"
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    model = resnet50(num_classes=num_classes, include_top=False)
    model.trainable = False
    model = tf.keras.Sequential([model,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])

    weight_path = "./5_ResNet/model/resnet50.ckpt"
    model.load_weights(weight_path)

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    print_res = "class:{}, prob: {:.3f}".format(class_dict[str(predict_class)], result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}, prob:{:.3f}".format(class_dict[str(i)], result[i]))
    plt.show()
    # 使用别人的预训练模型时， 要注意别人的预处理方式


if __name__ == '__main__':
    main()
