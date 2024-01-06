# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_lenet.py
# @Time       ：
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model_lenet import LeNet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 对图像进行预处理的函数
# Compose将使用的预处理方法打包为一个整体
transform = transforms.Compose(
    # ToTensor函数有两个功能：
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    [transforms.ToTensor(),
     # Normalize: 使用均值和标准差来Normalize操作
     # Normalize a tensor image with mean and standard deviation.、
     # 计算方法： output[channel] = (input[channel] - mean[channel]) / std[channel]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# 下载训练数据集 5万张
trainset = torchvision.datasets.CIFAR10(root='../data', download=False,
                                        train=True, transform=transform)
# 导入训练集，把它分成一个批次一个批次的，
# batch=32, 每一批随机抽32张图片，
# num_workers=0 载入数据的线程数， win环境下该参数要设置为0，
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=True, num_workers=0)
# 下载测试数据集1万张
testset = torchvision.datasets.CIFAR10(root="../data", train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, num_workers=0)
# 通过iter构建迭代器，next()函数就可以
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()
claeese = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(len(test_image))
print(len(test_label))

# 展示图片
# def show(img):
#     img = img / 2 + 0.5  # 反标准化，乘以0.5相当于除以2
#     npimg = img.numpy()
#     # 转化维度，ToTensor把图片变成[channel, height, width]
#     # 现需将其变成[height, width, channel]
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % claeese[test_label[i]] for i in range(4)))
# show(torchvision.utils.make_grid(test_image))

lenet = LeNet()
# 定义损失函数
# 这里解释一下为什么最后一个全连接层没有使用softmax函数，
# 点进去CrossEntropyLoss中可以看到它已经包含了LogSoftmax函数
loss_fuction = nn.CrossEntropyLoss()
# 定义优化器
# 第一个参数就是我们需要训练的参数， lenet.parameters()表示模型中所有可训练参数
optimizer = optim.Adam(lenet.parameters(), lr=0.0001)
# 训练过程
# EPOCHS表示要将训练迭代多少轮
EPOCHS = 5

for epoch in range(EPOCHS):
    # running_loss 累加训练过程中的损失
    running_loss = 0.0
    # 训练遍历训练集样本
    # enumerate函数， 可以返回每一批data， 以及对应的步数
    for step, data in enumerate(trainloader, start=0):
        # 将数据分离为输入以及对应的标签
        inputs, labels = data

        # 将历史损失梯度清零
        optimizer.zero_grad()

        outputs = lenet(inputs)
        loss = loss_fuction(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:
            # 表示在这个函数覆盖范围内，都不会再去计算梯度了
            with torch.no_grad():
                # 通过上面的next()函数已经将所有的测试集图片取出来了
                outputs = lenet(test_image)  # [batch, 10]
                # 第0个维度对应的是batch， 所以需要在第一个维度上计算最大值， 也就是在输出的10个节点中寻找最大的值
                # [1]表示我们只需要取出它的index, 而不需要它的最大值是多少
                predict_y = torch.max(outputs, dim=1)[1]
                # 将预测的标签类别与真实的标签类别进行比较， 在相同的地方就会返回True等于1， 不同的地方就是false也就是0
                # 通过sum求和函数就可知道在本地测试过程中预测对了多少个样本，
                # 前面计算得到的是tensor，通过item()取出它对应的数值， 拿到数值后再除以测试样本的数目就是准确率
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                print("[%d, %5d] train_loss: %.3f test_accuarcy: %.3f" % (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0
print("Finished Training")

# 最后保存模型
save_path = "./LeNet.pth"
torch.save(lenet.state_dict(), save_path)



















