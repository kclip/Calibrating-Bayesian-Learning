# Bayesian ResNet for ImageNet
# ResNet architecture ref:
# https://arxiv.org/abs/1512.03385
# Code adapted from torchvision package to build Bayesian model from deterministic model

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from variational_layers.conv_variational import Conv2dVariational
from variational_layers.linear_variational import LinearVariational

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

prior_mu = 0.0
prior_sigma = 0.1
posterior_mu_init = 0.0
posterior_rho_init = -2.0

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dVariational(prior_mu,
                             prior_sigma,
                             posterior_mu_init,
                             posterior_rho_init,
                             in_planes,
                             out_planes,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = self.relu(out)

        out, kl = self.conv2(out)
        kl_sum += kl
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, kl_sum


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       inplanes,
                                       planes,
                                       kernel_size=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       planes,
                                       planes,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       planes,
                                       planes * 4,
                                       kernel_size=1,
                                       bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dVariational(prior_mu,
                                  prior_sigma,
                                  posterior_mu_init,
                                  posterior_rho_init,
                                  inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)

        out, kl = self.conv2(out)
        kl_sum += kl
        out = self.bn2(out)
        out = F.relu(out)

        out, kl = self.conv3(out)
        kl_sum += kl
        out = self.bn3(out)

        if len(self.shortcut) > 0:
            out1, kl = self.shortcut[0](x)
            kl_sum += kl
            out += self.shortcut[1](out1)

        out = F.relu(out)


        #
        # out1, kl = self.shortcut[0](x)
        # kl_sum += kl
        # out += self.shortcut[1](out1)
        #
        # out = F.relu(out)

        return out, kl_sum

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)
        #
        # return out, kl_sum


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, temp=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       3,
                                       64,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(2)
        self.fc = LinearVariational(prior_mu, prior_sigma, posterior_mu_init,
                                    posterior_rho_init, 512 * block.expansion,
                                    num_classes)
        self.temp = temp

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    # def _make_layer(self, block, planes, blocks, stride):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes,
    #                       planes * block.expansion,
    #                       kernel_size=1,
    #                       stride=stride,
    #                       bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)

        for layer in self.layer1:
            if 'Variational' in str(layer):
                x, kl = layer(x)
                if kl is None:
                    kl_sum += kl
            else:
                x = layer(x)

        for layer in self.layer2:
            if 'Variational' in str(layer):
                x, kl = layer(x)
                if kl is None:
                    kl_sum += kl
            else:
                x = layer(x)

        for layer in self.layer3:
            if 'Variational' in str(layer):
                x, kl = layer(x)
                if kl is None:
                    kl_sum += kl
            else:
                x = layer(x)

        for layer in self.layer4:
            if 'Variational' in str(layer):
                x, kl = layer(x)
                if kl is None:
                    kl_sum += kl
            else:
                x = layer(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x, kl = self.fc(x)
        kl_sum += kl

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x, kl = self.fc(x)
        # kl_sum += kl

        return x, kl_sum


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(temp=1, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model