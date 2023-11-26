from torch import nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""
    def __init__(self, planes):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0))
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,  planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = F.relu(out)
        return out

if __name__ == '__main__':
    x = torch.rand(3, 3, 224, 224)
    model = ResBlock(3)
    y = model(x)
    print(y.shape)