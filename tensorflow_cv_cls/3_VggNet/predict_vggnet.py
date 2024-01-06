# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : predict_vggnet.py
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
from model_vggnet import vgg

def main():
    im_height = 224
    im_width = 224
    num_classes = 5

    # load image
    img_path = './tulip.jpg'

    img = Image.open(img_path)
    # resize
    img = img.resize((im_height, im_width))
    plt.imshow(img)

    # scaling pixel value to (0-1)
    img = np.array(img) / 225.

    # add image to a batch
    img = (np.expand_dims(img, 0))

    # read class_dict
    json_path = './3_VggNet/class_indices.json'

    with open(json_path, 'r') as f:
        class_dict = json.load(f)
    # create model
    model = vgg("vgg16", im_height=im_height, im_width=im_width, num_classes=num_classes)
    weights_path = "./3_VggNet/model/vggnet_1.h5"
    assert os.path.exists(img_path), "file: {} does not exist.".format(weights_path)
    model.load_weights(weights_path)

    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_re = "class: {}, prob: {:.3f}".\
        format(class_dict[str(predict_class)], result[predict_class])
    plt.title(print_re)
    for i in range(len(result)):
        print("class: {:10} prob: {:.3}".format(class_dict[str(i)], result[i]))
    plt.show()

if __name__ == '__main__':
    main()











