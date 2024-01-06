# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_alexnet.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model_alexnet import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os


def main1():
    data_root = "./data/flower_data"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    im_height = 224
    im_width = 224
    batch_size = 2
    epochs = 2

    # data_generator with data  augmentation
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        target_size=(im_height, im_width),
        class_mode="categorical"
    )
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((value, key) for key, value in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = val_image_generator.flow_from_directory(directory=val_dir,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
    total_val = val_data_gen.n
    print(f"using {total_train} images for training, {total_val} images for validation")

    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./alexnet.h5",
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor="val_loss")]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks
                        )
    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label="train_accuracy")
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()



def main2():
    data_root = "./data/flower_data"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    # data_generator with data  augmentation
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
    val_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        target_size=(im_height, im_width),
        class_mode="categorical"
    )
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((value, key) for key, value in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = val_image_generator.flow_from_directory(directory=val_dir,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
    total_val = val_data_gen.n
    print(f"using {total_train} images for training, {total_val} images for validation")

    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./alexnet.h5",
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor="val_loss")]

    history = model.fit_generator(generator=train_data_gen,
                                  steps_per_epoch=total_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_data_gen,
                                  validation_steps=total_val // batch_size,
                                  callbacks=callbacks)
    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.CategoricalCrossentropy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    best_test_loss = float('inf')
    for epoch in range(1, epochs+1):
        # clear history info
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)

        template = 'Epoch: {}, Loss: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))
        if test_loss.result() < best_test_loss:
            model.save_weights("./alexnet.ckpt", save_format="tf")


if __name__ == '__main__':
    main2()
