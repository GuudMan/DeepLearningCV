import json
from model_vgg import VggNet
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)), 
     transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

with open("./3_VggNet/class_indices.json", 'r+') as f:
    class_dict = json.load(f)


img_path = "./9433167170_fa056d3175.jpg"
input = Image.open(img_path)
input = data_transform(input)
# 添加一个batch维度
input = torch.unsqueeze(input, dim=0)

vggnet = VggNet()
vggnet.load_state_dict(torch.load('./3_VggNet/vggnet_best.pth'))
output = vggnet(input)
out_index = torch.max(output, dim=1)[1].item()
print(class_dict[str(out_index)])

