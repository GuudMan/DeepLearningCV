# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_googlenet.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model_googlenet import GoogleNet
import torch.utils.data


BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 0.00001
EPOCHS = 30

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                  )
    }
    image_path = "./data/flower_data"
    train_dir = os.path.join(image_path, "train")
    val_dir = os.path.join(image_path, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform['train'])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("./4_GoogleNet/class_indices.json", 'w') as json_file:
        json_file.write(json_str)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=NUM_WORKERS)

    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform['val'])
    val_num = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=NUM_WORKERS)

    print(f"using {train_num} images for training, using {val_num} images for validation")

    googlenet = GoogleNet(num_classes=5, aux_logits=False, init_weights=True)

    # 如果使用官方的预训练权重， 注意将权重载入官方的模型，不是我们自己实现的模型，
    # 官方的模型中使用了bn层以及改了一些参数， 不能混用
    # import torchvision
    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # # 预训练权重下载地址： https://download.pytorch.org/models/googlenet-1378be20.pth
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ['aux1.fc2.weight', 'aux1.fc2.bias',
    #             'aux2.fc2.weight', 'aux2.fc2.bias',
    #             'fc.weight', 'fc.bias']
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)
    #
    googlenet.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(googlenet.parameters(), lr=LEARNING_RATE)

    epochs = EPOCHS
    best_acc = 0.0
    save_path = "./4_GoogleNet/model/googlenet.pth"
    train_steps = len(train_dataloader)
    for epoch in range(epochs):
        # train
        googlenet.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            
            # logits, aux_logits2, aux_logits1 = googlenet(images.to(device))
            logits = googlenet(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            # loss1 = loss_function(aux_logits1, labels.to(device))
            # loss2 = loss_function(aux_logits2, labels.to(device))
            # loss = loss0 + loss1 + loss2
            loss = loss0
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch [{} / {}] loss: {:.3f}".format(epoch+1,
                                                                        epochs, loss)
        # validate
        googlenet.eval()
        # accumulate accurate number / epoch
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = googlenet(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" % 
              (epoch + 1, running_loss / train_steps, val_accurate))
        
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(googlenet.state_dict(), save_path)
    print("Finished Training")

if __name__ == '__main__':
    main()
        
