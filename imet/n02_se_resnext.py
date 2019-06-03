import math

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet as ResNetAvg7


class SEBlock(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SEBlock, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cSE = nn.Sequential(
            nn.Linear(planes, planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        cse = self.avgpool(x).view(b, c)
        cse = self.cSE(cse).view(b, c, 1, 1)
        cse = x * cse

        return cse


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, reduction=16, scse=None
    ):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if scse == "se":
            self.se = SEBlock(planes * 4, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if "se" in self._modules:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(ResNetBottleneck):
    """
    [SC][SE-]ResNeXt bottleneck type C
    """

    def __init__(
        self,
        inplanes,
        planes,
        baseWidth,
        cardinality,
        stride=1,
        downsample=None,
        reduction=16,
        scse=None,
    ):
        super(Bottleneck, self).__init__(
            inplanes, planes, stride, downsample, reduction, scse
        )

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(
            D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)


class SEBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        kwargs.pop("scse", None)
        super().__init__(*args, **kwargs, scse="se")


class ResNet(ResNetAvg7):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


class ResNeXt(ResNet):
    def __init__(self, block, baseWidth, cardinality, layers, num_classes=1103):
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        super(ResNeXt, self).__init__(block, layers, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                self.baseWidth,
                self.cardinality,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.baseWidth, self.cardinality)
            )

        return nn.Sequential(*layers)


def make_encoder(model):
    model.encoder = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    return model


def se_resnext50(**kwargs):
    """Constructs a SE-ResNeXt-50 model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 6, 3], **kwargs)
    model = make_encoder(model)
    return model


def se_resnext101(**kwargs):
    """Constructs a SE-ResNeXt-101 (32x4d) model."""
    model = ResNeXt(SEBottleneck, 4, 32, [3, 4, 23, 3], **kwargs)
    model = make_encoder(model)
    return model
