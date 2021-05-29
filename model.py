import torch
from torch import Tensor
import torch.nn as nn

NUM_CLASSES = 43
IMG_SIZE = 48
class cnn_basic(nn.Module):
    def __init__(self, inplane, outplane, k_size):
        super(cnn_basic, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, k_size, padding=((int)((k_size[0] - 1)/2), (int)((k_size[1] - 1)/2)))
        self.bn = nn.BatchNorm2d(outplane, eps=1e-06)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out



class cnn_branch(nn.Module):
    def __init__(self):
        super(cnn_branch, self).__init__()
        self.bs1 = cnn_basic(3, 32, (3, 3))
        self.bs2 = cnn_basic(32, 48, (7, 1))
        self.bs3 = cnn_basic(48, 48, (1, 7))
        self.mpool = nn.MaxPool2d((2,2))
        self.dp = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.bs1(x)
        out = self.bs2(out)
        out = self.bs3(out)
        out = self.mpool(out)

        out = self.dp(out)

        return out

class cnn_block(nn.Module):
    def __init__(self, k_size):
        super(cnn_block, self).__init__()
        self.bs1 = cnn_basic(48, 64, (k_size, 1))
        self.bs2 = cnn_basic(64, 64, (1, k_size))

    def forward(self, x):
        out = self.bs1(x)
        out = self.bs2(out)

        return out


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.branch1 = self._make_branch(3)
        self.branch2 = self._make_branch(7)

        self.mp1 = nn.MaxPool2d((2,2))
        self.dp1 = nn.Dropout(0.2)

        self.bs1 = cnn_basic(128, 128, (3, 3))
        self.bs2 = cnn_basic(128, 256, (3, 3))
        self.mp2 = nn.MaxPool2d((2,2))
        self.dp2 = nn.Dropout(0.3)
        self.fc = nn.Linear(9216, 256)
        self.bn = nn.BatchNorm1d(256, eps=1e-06)
        self.relu = nn.ReLU(True)
        self.dp3 = nn.Dropout(0.4)
        self.fcf = nn.Linear(256, NUM_CLASSES)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], 1)
        out = self.mp1(out)
        out = self.dp1(out)
        out = self.bs1(out)
        out = self.bs2(out)
        out = self.mp2(out)
        out = self.dp2(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dp3(out)
        out = self.fcf(out)

        return out


    def _make_branch(self, k_size):
        layers = []
        layers.append(cnn_branch())
        layers.append(cnn_block(k_size))
        
        return nn.Sequential(*layers)
