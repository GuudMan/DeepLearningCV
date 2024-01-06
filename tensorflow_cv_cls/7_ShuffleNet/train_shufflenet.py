# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_shufflenet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import math
import os
import sys
import json
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_shufflenet import shufflenet_v2_x1_0
from utils import generate_ds
import datetime


def main():
    data_root = "./data/flower_data/flower_photos"

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    learning_rate = 0.0003
    num_classes = 5

    log_dir = "./7_ShuffleNet/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, im_height, im_width, batch_size)

    # create model
    model = shufflenet_v2_x1_0(input_shape=(im_height, im_width, 3), num_classes=num_classes)

    # x1.0权重链接: https://pan.baidu.com/s/1M2mp98Si9eT9qT436DcdOw  密码: mhts
    pre_weights_path = "./7_ShuffleNet/model/pre_shufflenetv2_x1_0.h5"
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)
    model.summary()

    def scheduler(now_epoch):
        initial_lr = 0.1
        end_lr_rate = 0.1  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar("learning rate", data=new_lr, step=epoch)
        return new_lr

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
        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

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
            model.save_weights("./7_ShuffleNet/model/shufflenet_v2_x1_0.ckpt", save_format="tf")


if __name__ == '__main__':
    main()
