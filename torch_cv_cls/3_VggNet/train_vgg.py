import json
import sys
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from model_vgg import VggNet
from tqdm import tqdm

data_transforms = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), 
    "val": transforms.Compose({
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    })
}

BATCH_SIZE = 16
NUM_WORKERS = 8
LEARNING_RATE = 0.0001
EPOCHS = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = datasets.ImageFolder(root="./data/flower_data/train", 
                                 transform=data_transforms['train'])

flower_list = train_set.class_to_idx
cla_dict = dict((value, key) for key, value in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open("./3_VggNet/class_indices.json", 'w') as f:
    f.write(json_str)

train_num = len(train_set)
train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, 
                                           batch_size=BATCH_SIZE, 
                                           num_workers=NUM_WORKERS)

val_set = datasets.ImageFolder(root="./data/flower_data/val", 
                               transform=data_transforms['val'])
val_num = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, 
                                         shuffle=False, 
                                         batch_size=BATCH_SIZE, 
                                         num_workers=NUM_WORKERS)

# 实例化模型
vggnet = VggNet()
vggnet.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vggnet.parameters(), lr=LEARNING_RATE)

vgg_path = "./3_VggNet/vggnet_best.pth"
best_acc = 0.0
train_step = len(train_loader)
for epoch in range(EPOCHS):
    vggnet.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = vggnet(images.to(device))
        loss = loss_func(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, 
                                                                    EPOCHS, loss)

    # evaluate
    vggnet.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for step_val, val_data in enumerate(val_bar):
            val_images, val_labels = val_data
            val_output = vggnet(val_images.to(device))
            predict_y = torch.max(val_output, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_acc = acc / val_num
    print(f"train epochs: {epoch + 1}, train_loss: {running_loss / train_step}, val_acc: {val_acc}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(vggnet.state_dict(), vgg_path)
print("Finished Traiing!")






