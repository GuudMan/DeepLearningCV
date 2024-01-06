# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_alexnet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
from model_alexnet import AlexNet
import json

batch_size = 16
learning_rate = 0.0002
num_workers = 4
EPOCHS = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    'train': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    'val': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
}
train_img_path = "./data/flower_data/train"
val_img_path = "./data/flower_data/val"
train_set = datasets.ImageFolder(root=train_img_path, transform=data_transform["train"])
train_dataloader = torch.utils.data.DataLoader(train_set, shuffle=True,
                                               batch_size=batch_size, num_workers=num_workers)

val_set = datasets.ImageFolder(root=val_img_path, transform=data_transform["val"])
val_dataloader = torch.utils.data.DataLoader(val_set, shuffle=False,
                                               batch_size=batch_size, num_workers=num_workers)

flower_list = train_set.class_to_idx
cla_dict = dict((key, val) for val, key in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# show image
# def show_img(img):
#     img = img / 2 + 0.5  # 反标准化，乘以0.5相当于除以2
#     npimg = img.numpy()
#     # 转化维度，ToTensor把图片变成[channel, height, width]
#     # 现需将其变成[height, width, channel]
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
flower_list = train_set.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# print(flower_list)
val_img = iter(val_dataloader)
test_imgs, test_labels = val_img.next()
print(test_labels)
# show_img(utils.make_grid(img))
# print(' '.join('%5s' % cla_dict[test_labels[j].item()] for j in range(4)))

# 实例化模型， 定义损失函数， 定义优化器
alexnet = AlexNet()
alexnet.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=learning_rate)

best_acc = 0.0
for i in range(EPOCHS):
    train_loss = 0
    for step, data in enumerate(train_dataloader):
        inputs, labels = data

        # 历史梯度清零
        optimizer.zero_grad()
        # 计算输出
        output = alexnet(inputs.to(device))
        # 计算损失
        loss = loss_func(output, labels.to(device))
        # 反向传播
        loss.backward()
        # 优化器迭代
        optimizer.step()

        # 统计误差
        train_loss += loss.item()
        if step % 100 == 99:
            # 验证时不需要计算梯度
            with torch.no_grad():
                test_output = alexnet(test_imgs.to(device))  # [batch, 5]
                predict_y = torch.max(test_output, dim=1)[1]
                print("predict_y", predict_y)
                print("test_labels", test_labels)
                test_labels = test_labels.to(device)
                accuracy = (predict_y == test_labels).sum().item() / test_labels.size(0)
                print(f"[Epoch:{i+1}, step:{step + 1}], running_loss：{round(train_loss, 4)}, "
                      f"accuracy:{round(accuracy, 4)}")
                train_loss = 0.0
                if accuracy > best_acc:
                    best_acc = accuracy
                    model_alexnet = "./alexnet_best.pth"
                    torch.save(alexnet.state_dict(), model_alexnet)


print("Finish training")







