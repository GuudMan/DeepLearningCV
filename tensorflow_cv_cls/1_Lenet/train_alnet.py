from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model_lenet import LeNet
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # x_train shape (50000, 32, 32, 3)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 创建数据加载器
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # 创建模型
    lenet = LeNet()
    # 定义损失
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # 定义优化器
    optimizer = tf.keras.optimizers.Adam()

    # 定义训练损失和训练准确率
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    # define test loss and test accuracy
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    # define train function including calculating loss, applying gradient and calculating accuracy
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = lenet(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, lenet.trainable_variables)
        optimizer.apply_gradients(zip(gradients, lenet.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
    
    @tf.function
    def test_step(images, labels):
        predictions = lenet(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    
    EPOCHS = 5
    for epoch in range(EPOCHS):
        # 清除历史信息
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy:{}'
        print(template.format(epoch + 1, 
                              tf.round(train_loss.result(), 5), 
                              train_accuracy.result() * 100, 
                              test_loss.result(), 
                              test_accuracy.result()*100))


if __name__ == '__main__':
    main()
    