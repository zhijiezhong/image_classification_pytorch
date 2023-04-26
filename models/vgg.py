import torch
import torch.nn as nn

# 每个列表中的列表分别代表着 层数 输入通道 输出通道
from models.module.vgg_layers import make_layers, get_classifier

cfgs = {
    'vgg11': [[1, 3, 64], [1, 64, 128], [2, 128, 256], [2, 256, 512], [2, 512, 512]],
    'vgg13': [[2, 3, 64], [2, 64, 128], [2, 128, 256], [2, 256, 512], [2, 512, 512]],
    'vgg16': [[2, 3, 64], [2, 64, 128], [3, 128, 256], [3, 256, 512], [3, 512, 512]],
    'vgg19': [[2, 3, 64], [2, 64, 128], [4, 128, 256], [4, 256, 512], [4, 512, 512]]
}


class VGG(nn.Module):
    def __init__(self, feature, num_classes=2):
        super(VGG, self).__init__()
        self.feature = feature
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(1)
        self.classifier = get_classifier(num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def vgg(cfg: str, batch_norm: bool, num_classes: int):
    return VGG(make_layers(cfgs[cfg], batch_norm), num_classes)


def vgg11(num_classes=2):
    return vgg('vgg11', batch_norm=False, num_classes=num_classes)


def vgg11_bn(num_classes=2):
    return vgg('vgg11', batch_norm=True, num_classes=num_classes)


def vgg13(num_classes=2):
    return vgg('vgg13', batch_norm=False, num_classes=num_classes)


def vgg13_bn(num_classes=2):
    return vgg('vgg13', batch_norm=True, num_classes=num_classes)


def vgg16(num_classes=2):
    return vgg('vgg16', batch_norm=False, num_classes=num_classes)


def vgg16_bn(num_classes=2):
    return vgg('vgg16', batch_norm=True, num_classes=num_classes)


def vgg19(num_classes=2):
    return vgg('vgg19', batch_norm=False, num_classes=num_classes)


def vgg19_bn(num_classes=2):
    return vgg('vgg19', batch_norm=True, num_classes=num_classes)


if __name__ == '__main__':
    model = vgg19(1000)
    print(model)

    model2 = vgg19_bn(1000)
    print(model2)

    x = torch.rand(8, 3, 224, 224)
    print(model(x).shape)
    print(model2(x).shape)
