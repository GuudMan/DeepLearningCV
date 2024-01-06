# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : utils.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import json
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def read_split_data(root, val_rate=0.2):
    random.seed(0)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root)
                    if os.path.isdir(os.path.join(root, cla))]
    # 排序， 保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open("./6_MobileNet/class_indices.json", 'w') as json_file:
        json_file.write(json_str)

    # 存储训练，测试集的所有图片路径, 以及索引信息
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    # 存储每个类别的样本总数
    every_class_num = []
    supported = ['.jpg', '.JPG', '.JPEG', '.jpeg']
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历supported支持的所有文件路径 splittext会生成类似这样的('..path..\\5547758_eea9edfd54_n', '.jpg')
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本总数
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else: # 加入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print("{} images were found in dataset.\n{} for training, {} for validation".format(sum(every_class_num),
                                                                                        len(train_images_path),
                                                                                        len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每个类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align="center")
        # 将横坐标0， 1， 2， 3， 4替换成类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y = v + 5, s=str(v), ha="center")
        # 设置x坐标
        plt.xlabel("image class")
        # 设置y坐标
        plt.ylabel("number of images")
        # 设置柱状图标题
        plt.title("flower class distribution")
        plt.show()
    return train_images_path, train_images_label, val_images_path, val_images_label


def generate_ds(data_root,
                im_height,
                im_width,
                batch_size,
                val_rate:float = 0.1):
    """
    读取划分数据集， 并生成训练集和验证集的迭代器
    :param data_root: 数据根目录
    :param im_height:
    :param im_width:
    :param batch_size:
    :param val_rate:
    :return:
    """
    train_img_path, train_img_label, val_img_path, val_img_label = read_split_data(data_root, val_rate=val_rate)
    # AUTOTUNE 框架会根据可用的CPU自动设置最大的可用线程数，以使用多线程进行数据通道处理，将机器的算力拉满
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def process_train_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize_with_crop_or_pad(image, im_height, im_width)
        image = tf.image.random_flip_left_right(image)
        image = (image - 0.5) / 0.5
        return image, label

    def process_val_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize_with_crop_or_pad(image, im_height, im_width)
        image = tf.image.random_flip_left_right(image)
        image = (image - 0.5) / 0.5
        return image, label

    # configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size,
                                  shuffle=False):
        # 读取数据后缓存到内存
        ds = ds.cache()
        if shuffle:
            # 打乱数据顺序
            ds = ds.shuffle(buffer_size=shuffle_size)
        # 指定batch size
        ds = ds.batch(batch_size)
        # 在训练的同时准备下一个step的数据
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path),
                                                   tf.constant(train_img_label)))
    total_train = len(train_img_path)
    
    # use dataset.map to create a dataset of image, label pairs
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_path),
                                                   tf.constant(val_img_label)))
    total_val = len(val_img_path)
    # use Dataset.map to create a dataset of image, label pairs
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val)
    return train_ds, val_ds






