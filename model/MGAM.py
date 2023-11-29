from torch import nn
import torch

class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MGAM(nn.Module):
    def __init__(self, inplanes):
        super(MGAM, self).__init__()

        self.conv1 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(3, 1), stride=1, padding=(1, 0))
        ))
        self.conv2 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 3), stride=1, padding=(0, 1))
        ))
        self.conv3 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(5, 1), stride=1, padding=(2, 0))
        ))
        self.conv4 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 5), stride=1, padding=(0, 2))
        ))
        self.conv5 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(7, 1), stride=1, padding=(3, 0))
        ))
        self.conv6 = (nn.Sequential(
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            Conv2dBnRelu(inplanes, inplanes, kernel_size=(1, 7), stride=1, padding=(0, 3))
        ))
        self.conv7 = nn.Conv2d(inplanes*5, inplanes, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(True)


    def forward(self, x):
        x1 = self.conv1(x) + self.conv2(x)
        x2 = self.conv1(x+x1) + self.conv2(x+x1)
        x3 = self.conv3(x2+x) + self.conv4(x2+x)
        x4 = self.conv5(x+x3) + self.conv6(x+x3)
        x5 = x
        out = self.conv7(torch.cat((x1, x2, x3, x4, x5), dim=1))
        out = self.relu(self.bn(out+x))
        return out

if __name__ == '__main__':
    x = torch.rand(3, 3, 224, 224)
    model = MGAM(3)
    y = model(x)
    print(y.shape)