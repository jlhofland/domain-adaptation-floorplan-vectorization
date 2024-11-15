# Description: Residual block for the network. 
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, int(self.numOut / 2), bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(self.numOut / 2))
        self.conv2 = nn.Conv2d(int(self.numOut / 2), int(self.numOut / 2), bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(self.numOut / 2))
        self.conv3 = nn.Conv2d(int(self.numOut / 2), self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(
                self.numIn, self.numOut, bias=True, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual
