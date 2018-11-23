# import torch
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(4, 6, 5) # 4 channel in, 6 channels out, filter size 5
#         self.pool = torch.nn.MaxPool2d(2, 2) # 6 channel in, 6 channels out, filter size 2, stride 2
#         self.conv2 = torch.nn.Conv2d(6, 16, 5) # 6 channel in, 16 channels out, filter size 5
#         self.fc1 = torch.nn.Linear(16 * 125 * 125, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 28)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(torch.relu(x))
#         x = self.conv2(x)
#         x = self.pool(torch.relu(x))
#         x = x.view(-1, 16 * 125 * 125)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x


import torch
import torch.nn as nn
import torchvision

RESNET_ENCODERS = {
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}

class Net(nn.Module):
    def __init__(self, encoder_depth=101, pretrained=False, num_classes=28):
        super().__init__()

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        self.fc = nn.Linear(512 * 100 * (1 if encoder_depth == 34 else 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x