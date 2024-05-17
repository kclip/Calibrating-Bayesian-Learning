import torch.nn.functional as F
import torch
import torch.nn as nn
import torchbnn as bnn

# Define the Bayesian Wide ResNet Building Block

class Bayes_BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, prior_mu=0, prior_sigma=0.001):
        super(Bayes_BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu1(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        return torch.add(x, out)

# Define the Bayesian Wide ResNet
class Bayes_WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate=0.0, prior_mu=0, prior_sigma=0):
        super(Bayes_WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = Bayes_BasicBlock
        # 1st conv before any network block
        self.conv1 = bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=3, out_channels=nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = self._make_layer(block, nChannels[0], nChannels[1], n, stride=1, dropRate=dropRate)
        # 2nd block
        self.block2 = self._make_layer(block, nChannels[1], nChannels[2], n, stride=2, dropRate=dropRate)
        # 3rd block
        self.block3 = self._make_layer(block, nChannels[2], nChannels[3], n, stride=2, dropRate=dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=nChannels[3], out_features=100)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate=0.0):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def bayes_wideresnet_40_2(**kwargs):
    model = Bayes_WideResNet(40, 2, dropRate=0.0, prior_mu=0, prior_sigma=0.001, **kwargs)
    return model