# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_vggnet_GPU.py
# @Time       ：
# @Author     ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import matplotlib.pyplot as plt
from model_vggnet import vgg
import tensorflow as tf
import json
import os
import time
import glob
import random
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
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
    epochs = 50

    # class dict
    data_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    class_num = len(data_class)
    class_dict = dict((val, key) for key, val in enumerate(data_class))

    # reverse value and key of dict
    inverse_dict = dict((val, key) for key, val in class_dict.items())

    json_str = json.dumps(inverse_dict, indent=4)
    with open('./3_VggNet/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # load train image list
    train_image_list = glob.glob(train_dir + "/*/*.jpg")
    random.shuffle(train_image_list)
    train_num = len(train_image_list)

    train_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in train_image_list]
    print(train_label_list)

    # load validation images list
    val_image_list = glob.glob(val_dir + "/*/*.jpg")
    random.shuffle(val_image_list)
    val_num = len(val_image_list)

    val_label_list = [class_dict[path.split(os.path.sep)[-2]] for path in val_image_list]

    print("using {} images for training, {} images for validation".format(train_num, val_num))

    def process_path(img_path, label):
        label = tf.one_hot(label, depth=class_num)
        print("label", label)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [im_height, im_width])
        return image, label

    # 表示tf.data模块运行时，框架会根据可用的CPU自动设置最大的可用线程数，
    # 以使用多线程进行数据通道处理， 将机器的算力拉满， 注意返回的变量是个常数， 表示可用的线程数
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # load tarin dataset
    # preftch(): 开启预加载数据，使得在 GPU 训练的同时 CPU 可以准备数据
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
    train_dataset = train_dataset.shuffle(buffer_size=train_num).\
        map(process_path, num_parallel_calls=AUTOTUNE).repeat().\
        batch(batch_size).prefetch(AUTOTUNE)

    # load val dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list, val_label_list))
    val_dataset = val_dataset.map(process_path, num_parallel_calls=
    AUTOTUNE).repeat().batch(batch_size)

    # 实例化模型
    model = vgg("vgg16", 224, 224, 5)
    model.summary()

    # using keras low api for training
    # from_logits = False 即输出层是带softmax激活函数的
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    # train_loss = tf.keras.metrics.Mean(name="train_loss")
    # train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name="train_accuracy")

    # test_loss = tf.keras.metrics.Mean(name="test_loss")
    # test_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='test_accuracy')

    # @tf.function
    # def train_step(images, labels):
    #     """
    #     tensorflow 提供tf.GradientTape api来实现自动求导功能。
    #     只要在tf.GradientTape()上下文中执行的操作，都会被记录
    #     与“tape”中，然后tensorflow使用反向自动微分来计算相关操作的梯度。
    #     """
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #     train_loss(loss)
    #     train_accuracy(labels, predictions)

    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)

    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)

    # best_test_loss = float('inf')
    # train_step_num = train_num // batch_size
    # val_step_num = val_num // batch_size

    # for epoch in range(1, epochs + 1):
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    #     test_loss.reset_states()
    #     test_accuracy.reset_states()

    #     t1 = time.perf_counter()
    #     for index, (images, labels) in enumerate(train_dataset):
    #         train_step(images, labels)
    #         if index + 1 == train_step_num:
    #             break
    #     print(time.perf_counter() - t1)

    #     for index, (images, labels) in enumerate(val_dataset):
    #         test_step(images, labels)
    #         if index + 1 == val_step_num:
    #             break

    #     template = "Epoch {}, Loss:{}, Accuracy:{}, Test Loss:{}, Test Accuracy:{}"
    #     print(template.format(epoch, train_loss.result(),
    #                           train_accuracy.result(),
    #                           test_loss.result(),
    #                           test_accuracy.result()))
    #     if test_loss.result() < best_test_loss:
    #         model.save_weights("./3_VggNet/model/vggnet.ckpt".format(epoch), save_format='tf')
    

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./3_VggNet/model/vggnet_{epoch}.h5",
                                                    save_best_only=True, 
                                                    save_weights_only=True, 
                                                    monitor='val_loss')]
    history = model.fit(x=train_dataset, 
                        steps_per_epoch=train_num // batch_size, 
                        epochs=epochs, 
                        validation_data=val_dataset, 
                        validation_steps=val_num // batch_size, 
                        callbacks=callbacks)
      

if __name__ == '__main__':
    main()


























