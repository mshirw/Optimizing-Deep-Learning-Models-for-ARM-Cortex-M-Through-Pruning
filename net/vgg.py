import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.prune import PruningModule, MaskedLinear

defaultcfg = {
    7: [8, 'M', 16, 'M', 32, 32, 'M', 40, 40],
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


# class VGG(PruningModule):
class VGG(nn.Module):
    def __init__(self, depth=16, bn=True, mask=False):
        super(VGG, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        cfg = defaultcfg[depth]
        self.cfg = cfg
        self.feature = self.make_layers(cfg, bn)
        num_classes = 10
        # print(cfg[-1])
        
        self.classifier = nn.Sequential(
            # nn.Linear(cfg[-1] * 7 * 7, 4096),
            # nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes)
            # nn.Linear(160, num_classes, bias=False)
            linear(160, num_classes, bias=False)
        )
        
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.quant(x)
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        # y = self.dequant(x)

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()
            elif isinstance(m, MaskedLinear):
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()
    
    def fuse_model(self):
        for i in range(15):
            if isinstance(self.feature[i], nn.Conv2d):
                self.feature = torch.quantization.fuse_modules(self.feature, [str(i), str(i+1)])