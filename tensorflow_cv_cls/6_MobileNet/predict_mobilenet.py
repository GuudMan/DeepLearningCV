# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_mobilenet.py
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
from model_mobilenet_v2 import MobileNetV2


def main():
    im_height = 224
    im_width = 224
    num_classes = 5

    img_path = "./tulip.jpg"
    img = Image.open(img_path)
    img = img.resize((im_height, im_width))

    # scaling pixel value to (0-1)
    img = np.array(img).astype(np.float32)
    img = ((img / 255.) - 0.5) * 2.0

    # add a batch
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = "./6_MobileNet/class_indices.json"
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    model = MobileNetV2(include_top=False)
    model.trainable = False
    model = tf.keras.Sequential([model,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(num_classes, activation="relu"),
                                 tf.keras.layers.Softmax()])

    weight_path = "./6_MobileNet/model/mobilenetv2.ckpt"
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