import torch
import torch.nn as nn
import torch.nn.functional as F

# 先搭建vgg19
class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        # (224 - 3 + 2*1)/1 + 1 = 224 -> (224, 224, 64)
        self.conv1_64_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        # (223 - 3 + 2*1)1 + 1 = 224 -> (224, 224, 64)
        self.conv1_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        # (224, 224, 64) -> (112, 112, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (112 - 3 + 2*1)/1 + 1 = 112  (112, 112, 64) -> (112, 112, 128)
        self.conv2_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # (112, 128, 128) -> (56, 56, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (56 - 3 + 2 * 1)/ 1 + 1 = 56 (56, 56, 128) -> (56, 56, 256)
        self.conv3_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_256_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # maxpool  (56, 56, 256) -> (28, 28, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_512_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_512_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # maxpool (28, 28, 512) -> (14, 14, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (14, 14, 512)
        self.conv5_512_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_512_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_512_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # maxpool (14, 14, 512) -> (7, 7, 512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=5)

    def forward(self, x):
        x = self.conv1_64_1(x)
        x = self.conv1_64_2(x)
        x = self.maxpool1(x)

        x = self.conv2_128_1(x)
        x = self.conv2_128_2(x)
        x = self.maxpool2(x)

        x = self.conv3_256_1(x)
        x = self.conv3_256_2(x)
        x = self.conv3_256_3(x)
        x = self.maxpool3(x)

        x = self.conv4_512_1(x)
        x = self.conv4_512_2(x)
        x = self.conv4_512_3(x)
        x = self.maxpool4(x)

        x = self.conv5_512_1(x)
        x = self.conv5_512_2(x)
        x = self.conv5_512_3(x)
        x = self.maxpool5(x)  # [batch, c, h, w] -> [-1, c*h*w]

        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# input = torch.rand([8, 3, 224, 224])
# vggnet = VggNet()
# print(vggnet)
# output = vggnet(input)
# print(output)