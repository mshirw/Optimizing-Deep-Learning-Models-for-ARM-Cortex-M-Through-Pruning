import torch.nn as nn
import torch
import torch.nn.functional as F
from net.prune import PruningModule, MaskedLinear

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, mask=False):
        super(AlexNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            linear(256, num_classes, bias=False)
            # nn.Linear(128, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.feature(x)
        # x = x.view(x.size(0), 256 * 2 * 2)
        x = nn.AvgPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x