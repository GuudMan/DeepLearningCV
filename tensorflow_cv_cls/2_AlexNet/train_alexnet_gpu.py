# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_AlexNet_GPU.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import tensorflow as tf
import matplotlib.pyplot as plt
from model_alexnet import AlexNet_v1, AlexNet_v2
import json
import os
import time
import glob
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)

    data_root = "./data/flower_data"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    # class dict
    data_class = [cla for cla in os.listdir(train_dir)
                  if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    class_dict = dict((value, index) for index, value in enumerate(data_class))

    # reverse value and key of dict
    inverse_dict = dict((val, key) for key, val in class_dict.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open("./class_indices.json", 'w') as json_file:
        json_file.write(json_str)

    # load train images list
    train_image_list = glob.glob(train_dir + "/*/*.jpg")
    random.shuffle(train_image_list)
    train_num = len(train_image_list)
    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]
    print(train_label_list)

    val_image_list = glob.glob(val_dir + "/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)

    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]
    print("using {} for images for training, {} image for validation.".format(train_num, val_num))

    def process_path(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        return image, label
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num)\
        .map(process_path, num_parallel_calls=AUTOTUNE)\
        .repeat().batch(batch_size).prefetch(AUTOTUNE)

    # load train dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(batch_size)

    # 实例化模型
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

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
    train_step_num = train_num // batch_size
    val_step_num = val_num // batch_size
    
    for epoch in range(1, epochs+1):
        # clear history info
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        t1 = time.perf_counter()
        for index, (imags, labels) in enumerate(train_dataset):
            train_step(imags, labels)
            if index + 1 == train_step_num:
                break
        print(time.perf_counter() - t1)
        
        for index, (images, labels)  in enumerate(val_dataset):
            test_step(images, labels)
            if index + 1 == val_step_num:
                break
            
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch, 
                              train_loss.result(), 
                              train_accuracy.result()*100, 
                              test_loss.result(), 
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            model.save_weights("alexnet_gpu.ckpt".format(epoch), save_format='tf')
            

if __name__ == '__main__':
    main()










