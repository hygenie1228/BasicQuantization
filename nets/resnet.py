import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import Q_Conv2d, Q_Linear, Q_ReLU

def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1):
    return Q_Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=7, stride=stride,
                    padding=padding, groups=groups, bias=False, dilation=padding)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, padding: int = 1):
    return Q_Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                    padding=padding, groups=groups, bias=False, dilation=padding)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return Q_Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, 
                    stride=stride, bias=False)

def linear(in_channels: int, out_channels: int):
    return Q_Linear(in_channels=in_channels, out_channels=out_channels)

def relu():
    return Q_ReLU()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            in_planes, planes, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = relu()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = relu()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, stem, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if stem == 'cifar':
            self.conv1 = conv3x3(3, 64, stride=1, padding=1)
            #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif stem == 'imagenet':
            self.conv1 = conv7x7(3, 64, stride=2, padding=3)
            
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = relu()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = linear(512*block.expansion, num_classes)
        #self.linear = linear(512*block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def resnet18(stem='cifar', num_classes=10):
    return ResNet(stem, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(stem='cifar', num_classes=10):
    return ResNet(stem, BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(stem='cifar', num_classes=10):
    return ResNet(stem, Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(stem='cifar', num_classes=10):
    return ResNet(stem, Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(stem='cifar', num_classes=10):
    return ResNet(stem, Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
