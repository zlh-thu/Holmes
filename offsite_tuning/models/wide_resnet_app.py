import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .cbn import CleanBatchNorm2d
from collections import OrderedDict

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



class wide_basic_app(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic_app, self).__init__()
        self.bn1 = CleanBatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = CleanBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    '''def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out'''

    def forward(self, x, mask=None):

        out = self.bn1(x, mask)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.bn2(out, mask)
        out = F.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

    def _shortcut(self, input, mask):
        if len(self.shortcut)==0:
            return self.shortcut(input)
        else:
            return self.shortcut[1](self.shortcut[0](input), mask)

# 28, 10, 0.3, 10
class Wide_ResNet_APP(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet_APP, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic_app, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic_app, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic_app, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = CleanBatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, mask=None, lin=0, lout=6):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out, mask))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



def through_layer(layer, x, mask=None):
    for i in range(len(layer)):
        if isinstance(layer[i], CleanBatchNorm2d) or isinstance(layer[i], wide_basic_app):
            x = layer[i](x, mask)
        else:
            x = layer[i](x)
    return x


if __name__ == '__main__':
    net=Wide_ResNet_APP(28, 10, 0.3, 10)
    img = torch.randn(5, 3, 32, 32)
    mask = torch.ones((5,))
    mask[-1] = 0
    y = net(img, mask)

    print(y.size())