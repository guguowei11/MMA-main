from torch import nn
import torch

class Spatial_Attention(nn.Module):
    """空间注意力模块"""
    def __init__(self,kernel_size = 7):
        super(Spatial_Attention, self).__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size ,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Channel_Attention(nn.Module):
    def __init__(self,in_ch,ration = 16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch,in_ch//ration,kernel_size=1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch//ration,in_ch,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class AGFM(nn.Module):
    def __init__(self, in_ch):
        super(AGFM, self).__init__()
        self.sa = Spatial_Attention()
        self.ca = Channel_Attention(in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2,in_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.ones(1),requires_grad=True)
        self.bata = nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self, x):
        x1 = self.ca(x)
        x1 = torch.mul(x,x1)
        x2 = self.sa(x1+self.gamma*x)
        x2 = torch.mul(self.bata*x1,x2)
        x3 = self.conv(torch.cat((x2, x), dim=1))
        out = x + x3

        return out

if __name__ == '__main__':
    x = torch.rand(3, 64, 224, 224)
    model = AGFM(64)
    y = model(x)
    print(y.shape)