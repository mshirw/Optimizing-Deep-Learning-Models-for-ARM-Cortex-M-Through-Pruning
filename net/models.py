import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)
        

    def forward(self, x):
        
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        # print(x.size)
        # x = x.view(-1, 120)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class cifar10_net(PruningModule):
    def __init__(self, mask=False):
        super(cifar10_net, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = linear(576, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2)

        # Conv2
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=(3, 3), stride=2)

        #Conv3
        x = self.conv3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=(3, 3), stride=2)
        # Fully-connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        return x


class LeNet_5_onnx(nn.Module):
    def __init__(self, mask=False):
        super(LeNet_5_onnx, self).__init__()
        # linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), padding=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        # print("conv1_shape:{}".format(x.size()))
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        # print("maxpool1_shape:{}".format(x.size()))

        # Conv2
        x = self.conv2(x)
        # x = self.bn2(x)
        # print("conv2_shape:{}".format(x.size()))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=3)
        # print("maxpool2_shape:{}".format(x.size()))

        # Fully-connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # print(self.fc1.weight.requires_grad)
        x = F.log_softmax(x, dim=1)
        
        return x

class LeNet_5_3x3(nn.Module):
    def __init__(self, mask=False):
        super(LeNet_5_3x3, self).__init__()
        # linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 10, bias=False)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)        
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=3)

        # Fully-connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)        
        x = F.log_softmax(x, dim=1)
        
        return x