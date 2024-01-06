# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_alexnet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model_alexnet import AlexNet_v1, AlexNet_v2

def main():
    im_height = 224
    im_width = 224

    # load image
    img_path = './2_AlexNet/tulip.jpg'
    img = Image.open(img_path)
    img = img.resize((im_height, im_width))
    plt.imshow(img)

    # scaling pixel value to(0-1)
    img = np.array(img) / 255

    # add image to a batch where it's the only member
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    
    with open(json_path, 'r') as f:
        class_dict = json.load(f)
    
    # create model
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    weight_path = "./2_AlexNet/alexnet.h5"
    model.load_weights(weight_path)
    
    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    
    print_res = "class: {}, prob: {}".format(class_dict[str(predict_class)], 
                                             result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10} prob {:.3}".format(class_dict[str(i)], result[i]))
    plt.show()


if __name__ == '__main__':
    main()


