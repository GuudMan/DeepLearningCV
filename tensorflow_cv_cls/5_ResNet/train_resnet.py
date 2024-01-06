# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_googlenet.py
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
from model_resnet import resnet50


def main():
    data_root = "./data/flower_data"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    learning_rate = 0.0003
    num_classes = 5

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94


    def pre_function(img):
        # img = img / 255.
        # img = (img - 0.5) / 0.5
        # return img
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
        return img


    train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                               horizontal_flip=True)
    val_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode="categorical")
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    json_str = json.dumps(inverse_dict, indent=4)
    # write dict into json file
    with open("./5_ResNet/class_indices.json", 'w') as f:
        f.write(json_str)

    val_data_gen = val_image_generator.flow_from_directory(directory=val_dir, shuffle=False,
                                                           batch_size=batch_size, target_size=(im_height, im_width),
                                                           class_mode="categorical")
    total_val = val_data_gen.n
    print(f"using {total_train} for training, using {total_val} for validation")

    model = resnet50(num_classes=5, include_top=False)
    # model.summary()

    # 直接下载转好的权重
    # download weights : 链接: https://pan.baidu.com/s/1tLe9ahTMIwQAX7do_S59Zg  密码: u199
    pre_weights_path = "./5_ResNet/model/tf_resnet50_weights/pretrain_weights.ckpt"
    model.load_weights(pre_weights_path)
    model.trainable = False
    model.summary()

    model = tf.keras.Sequential([model,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])

    # using keras low api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        output = model(images, training=False)
        loss = loss_object(labels, output)

        val_loss(loss)
        val_accuracy(labels, output)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # train
        train_bar = tqdm(range(total_train // batch_size), file=sys.stdout)
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1, epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())
        # validation
        val_bar = tqdm(range(total_val // batch_size), file=sys.stdout)
        for step in val_bar:
            val_images, val_labels = next(val_data_gen)
            val_step(val_images, val_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # only save best weights
        if val_accuracy.result() > best_acc:
            best_acc = val_accuracy.result()
            model.save_weights("./5_ResNet/model/resnet50.ckpt", save_format="tf")


if __name__ == '__main__':
    main()
