import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 6, 5) # 4 channel in, 6 channels out, filter size 5
        self.pool = torch.nn.MaxPool2d(2, 2) # 6 channel in, 6 channels out, filter size 2, stride 2
        self.conv2 = torch.nn.Conv2d(6, 16, 5) # 6 channel in, 16 channels out, filter size 5
        self.fc1 = torch.nn.Linear(16 * 125 * 125, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 28)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))
        x = x.view(-1, 16 * 125 * 125)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Resnet(nn.Module):
    def __init__(self, encoder_depth=101, pretrained=False):
        super().__init__()

        RESNET_ENCODERS = {
            34: torchvision.models.resnet34,
            50: torchvision.models.resnet50,
            101: torchvision.models.resnet101,
            152: torchvision.models.resnet152,
        }

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)
        print("ResNet" + str(encoder_depth) + " is used.")

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        add_channel = torch.empty(64, 1, 7, 7)
        init.xavier_normal_(add_channel)
        # init.constant_(add_channel, 0.01)

        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w, add_channel), dim=1))
        # self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        self.fc = nn.Linear(512 * 100 * (1 if encoder_depth == 34 else 4), 28)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    # init.constant_(m.bias, 0.01)

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

class BCNN(nn.Module):
    def __init__(self, encoder_depth=101, pretrained=False):
        super().__init__()

        RESNET_ENCODERS = {
            34: torchvision.models.resnet34,
            50: torchvision.models.resnet50,
            101: torchvision.models.resnet101,
            152: torchvision.models.resnet152,
        }

        encoder = RESNET_ENCODERS[encoder_depth](pretrained=pretrained)
        print("ResNet" + str(encoder_depth) + " is used.")

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        add_channel = torch.empty(64, 1, 7, 7)
        init.xavier_normal_(add_channel)
        # init.constant_(add_channel, 0.01)

        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w, add_channel), dim=1))
        # self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        # self.fc = nn.Linear(512 * 100 * (1 if encoder_depth == 34 else 4), 28)
        self.fc = nn.Linear(2048**2, 28)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    # init.constant_(m.bias, 0.01)

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

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # BCNN
        N = x.size()[0]
        assert x.size() == (N, 2048, 10, 10)
        x = x.view(N, 2048, 10**2)
        x = torch.bmm(x, torch.transpose(x, 1, 2))  # Bilinear
        assert x.size() == (N, 2048, 2048)
        x = x.view(N, 2048**2)
        x = torch.sqrt(x + 1e-5)                # L2
        x = nn.functional.normalize(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Densenet(nn.Module):
    def __init__(self, encoder_depth=169, pretrained=False):
        super().__init__()

        DENSENET_ENCODERS = {
            121: torchvision.models.densenet121,
            161: torchvision.models.densenet161,
            169: torchvision.models.densenet169,
            201: torchvision.models.densenet201,
        }

        encoder = DENSENET_ENCODERS[encoder_depth](pretrained=pretrained).features
        print("DenseNet" + str(encoder_depth) + " is used.")

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if encoder_depth == 161:
            add_channel = torch.empty(96, 1, 7, 7)
        else:
            add_channel = torch.empty(64, 1, 7, 7)
        init.kaiming_normal_(add_channel)
        # init.constant_(add_channel, 0.01)

        w = encoder.conv0.weight
        if encoder_depth == 161:
            self.conv0 = nn.Conv2d(4, 96, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0.weight = nn.Parameter(torch.cat((w, add_channel), dim=1))
        self.norm0 = encoder.norm0
        self.relu0= nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.denseblock1 = encoder.denseblock1
        self.transition1 = encoder.transition1
        self.denseblock2 = encoder.denseblock2
        self.transition2 = encoder.transition2
        self.denseblock3 = encoder.denseblock3
        self.transition3 = encoder.transition3
        self.denseblock4 = encoder.denseblock4

        self.norm5 = encoder.norm5
        self.fc = nn.Linear(565248, 28)
        # self.fc = nn.Linear(425984, 28)
        if encoder_depth == 121:
            fc_para = 1024
        elif encoder_depth == 161:
            fc_para = 2208
        elif encoder_depth == 169:
            fc_para = 1664
        elif encoder_depth == 201:
            fc_para = 1920
        self.fc = nn.Linear(16 * 16 * fc_para, 28)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal(m.weight)
                    # init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.denseblock4(x)

        x = self.norm5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Inception(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.encoder = torchvision.models.inception_v3(num_classes=28, aux_logits=False, pretrained=False)

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

        # add_channel = torch.empty(32, 1, 3, 3)
        # init.xavier_normal_(add_channel)
        # w = self.encoder.Conv2d_1a_3x3.conv.weight
        # self.encoder.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        # self.encoder.Conv2d_1a_3x3.conv.weight = nn.Parameter(torch.cat((w, add_channel), dim=1))
        # self.encoder.Conv2d_1a_3x3.bn = self.encoder.Conv2d_1a_3x3.bn

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.encoder(x)

        return x

class Squeezenet(nn.Module):
    def __init__(self, encoder_depth=10, pretrained=False):
        super().__init__()

        SQUEEZENET_ENCODERS = {
            10: torchvision.models.squeezenet1_0,
            11: torchvision.models.squeezenet1_1,
        }

        encoder = SQUEEZENET_ENCODERS[encoder_depth](pretrained=pretrained).features

        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with zeros
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        add_channel = torch.empty(64, 1, 7, 7)
        init.xavier_normal_(add_channel, gain=init.calculate_gain('relu'))
        # init.constant_(add_channel, 0.01)

        w = encoder.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w, add_channel), dim=1))
        # self.conv1.weight = nn.Parameter(torch.cat((w, torch.zeros(64, 1, 7, 7)), dim=1))

        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.avgpool = encoder.avgpool
        # self.fc = nn.Linear(512 * 100 * (1 if encoder_depth == 34 else 4), 28)
        self.fc = nn.Linear(2048**2, 28)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    # init.constant_(m.bias, 0.01)

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

        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # BCNN
        N = x.size()[0]
        assert x.size() == (N, 2048, 10, 10)
        x = x.view(N, 2048, 10**2)
        x = torch.bmm(x, torch.transpose(x, 1, 2))  # Bilinear
        assert x.size() == (N, 2048, 2048)
        x = x.view(N, 2048**2)
        x = torch.sqrt(x + 1e-5)                # L2
        x = nn.functional.normalize(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x