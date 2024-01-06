# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : batch_predict_resnet.py
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

import numpy as np
import tensorflow as tf
from PIL import Image
from model_resnet import resnet50

def main():
    im_height = 224
    im_width = 224
    classes_num = 5
    batch_size = 32

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    img_root = "./data/flower_data/val/roses"
    img_list = [os.path.join(img_root, i) for i in os.listdir(img_root) if i.endswith(".jpg")]

    json_path = "./5_ResNet/class_indices.json"

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)

    # create model
    feature = resnet50(num_classes=classes_num, include_top=False)
    feature.trainable = False
    model = tf.keras.Sequential([
        feature,
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes_num),
        tf.keras.layers.Softmax()
    ])
    weight_path = "./5_ResNet/model/resnet50.ckpt"
    model.load_weights(weight_path)

    # 每次预测时将多张图片导包成一个batch
    for ids in range(0, len(img_list) // batch_size):
        img_list_ids = []
        for img_i in img_list[ids * batch_size: (ids + 1) * batch_size]:
            img = Image.open(img_i)
            img = img.resize((im_height, im_width))
            img = np.array(img).astype(np.float32)
            img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
            img_list_ids.append(img)

        # batch img
        batch_img = np.stack(img_list_ids, axis=0)

        # predict
        result = model.predict(batch_img)
        predict_classes = np.argmax(result, axis=1)

        for idx, class_index in enumerate(predict_classes):
            print_res = "image : {} class: {} prob: {:.3f}".format(img_list[ids * batch_size + idx], 
                                                                   class_dict[str(class_index)], 
                                                                   result[idx][class_index])
            print(print_res)

if __name__ == '__main__':
    main()
