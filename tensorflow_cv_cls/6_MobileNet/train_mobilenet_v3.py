# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_mobilenetv3.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import sys
import json
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_mobilenet_v3 import mobilenet_v3_large
from utils import generate_ds

def main():
    data_root = "./data/flower_data/flower_photos"

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 20
    learning_rate = 0.0003
    num_classes = 5
    freeze_layer = False

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, im_height, im_width, batch_size)

    # create model
    model = mobilenet_v3_large(input_shape=(im_height, im_width, 3), num_classes=num_classes, include_top=True)
    # load weights 链接： https://pan.baidu.com/s/13uJznKeqHkjUp72G_gxe8Q  密码： 8quu
    pre_weights_path = "./6_MobileNet/model/weights_mobilenet_v3_large_224_1.0_float.h5"
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    if freeze_layer:
        # freeze layer, only training 2 last layers
        for layer in model.layers:
            if layer.name not in ["Conv_2", "Logits/Conv2d_1c_1x1"]:
                layer.trainable=False
            else:
                print("training: " + layer.name)
    model.summary()

    # using keras low api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1, epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())
        # validation
        val_bar = tqdm(val_ds, file=sys.stdout)
        for val_images, val_labels in val_bar:
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # only save best weights
        if val_accuracy.result() > best_acc:
            best_acc = val_accuracy.result()
            model.save_weights("./6_MobileNet/model/mobilenetv3_large.ckpt", save_format="tf")


if __name__ == '__main__':
    main()
