from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M

from imet.n01_utils import ON_KAGGLE
from imet.n02_se_resnext import se_resnext50
from imet.n02_cadene import se_resnext101_32x4d as se_resnext101
from imet.n02_effnet import effnet_b3
from imet.n04_dataset import N_CLASSES


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f"../input/{model_name}/{model_name}.pth"
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
    ):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(), nn.Linear(self.net.fc.in_features, num_classes)
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)


    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


def densenet161():
    model = M.densenet161(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
    model.encoder = model.features
    return model


if __name__ == '__main__':
    model = densenet161()
    checkpoint = torch.load(
        '/media/n01z3/red3_2/learning_dumps/dn161_f0_a6_e/fold_1/stage2/weights/epoch80_metric0.60313.pth')
    print(checkpoint.keys())

    loaded_state_dict = checkpoint['state_dict']
    sanitized = dict()
    for key in checkpoint['state_dict'].keys():
        sanitized[key.replace('classifier.1', 'classifier')] = loaded_state_dict[key]
        if 'classifier' in key:
            print(key)


    print(sanitized['classifier.weight'].shape)
    print(sanitized['classifier.bias'].shape)

    fc_weights = torch.from_numpy(sanitized['classifier.weight'].data.cpu().numpy()[:-1, :])
    fc_bias = torch.from_numpy(sanitized['classifier.bias'].data.cpu().numpy()[:-1])

    sanitized['classifier.weight'] = fc_weights
    sanitized['classifier.bias'] = fc_bias

    torch.save(sanitized, '../weights/dn161.pth')

    model.load_state_dict(torch.load('../weights/dn161.pth'))
    images = torch.autograd.Variable(torch.randn(4, 3, 320, 320))
    # if torch.cuda.is_available():
    #     images = images.cuda()

    out = model.forward(images)
    print(out.shape)
