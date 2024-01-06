# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : train_mobilenet.py
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
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.utils.data
from model_mobilenetv2 import MobileNetV2

BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 0.00001
EPOCHS = 10


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])]
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
    with open("./6_MobileNet/class_indices.json", 'w') as json_file:
        json_file.write(json_str)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=NUM_WORKERS)

    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform['val'])
    val_num = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUM_WORKERS)

    print(f"using {train_num} images for training, using {val_num} images for validation")

    # 因为加载预训练模型，所以这里的class_num使用模型中默认，也就是1000
    net = MobileNetV2(num_classes=5)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./6_MobileNet/model/pre_mobilenet_v2.pth"
    pre_weights = torch.load(model_weight_path, map_location="cpu")

    # delete classifier weights
    pre_dict = {k:v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    epochs = EPOCHS
    best_acc = 0.0
    save_path = "./6_MobileNet/model/mobilenetv2.pth"
    train_steps = len(train_dataloader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()

            # logits, aux_logits2, aux_logits1 = googlenet(images.to(device))
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch [{} / {}] loss: {:.3f}".format(epoch + 1,
                                                                         epochs, loss)
        # validate
        net.eval()
        # accumulate accurate number / epoch
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Finished Training")


if __name__ == '__main__':
    main()