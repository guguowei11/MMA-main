import math
import numpy as np
import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
import torch.nn.functional as F
from MSP import MSP
from ConvStage import ResBlock
from TransformerStage import PVTEncoder


class Conv2dBN(nn.Module):
    """Convolution with BN module."""
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
        bn_weight_init=1, norm_layer=nn.BatchNorm2d, act_layer=None):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class PVT_CNN_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `PVT_CNNEncoder` layers."""
    def __init__(self, embed_dim, num_layers=1, num_heads=8, mlp_ratio=3, num_path=4, sr_ratios=8, drop_path_list=[]):
        super().__init__()

        self.PVT_blks = nn.ModuleList([
            PVTEncoder(
                embed_dim,
                num_layers,
                num_heads,
                mlp_ratio,
                sr_ratios=sr_ratios,
                drop_path_list=drop_path_list,
            ) for _ in range(num_path)])

        self.InvRes = self._make_layer(ResBlock, embed_dim, num_layers)
        self.aggregate = Conv2dBN(embed_dim * (num_path + 1), embed_dim)

    def _make_layer(self, block, planes, num_blocks):
            num_blocks = [num_blocks]
            layers = []
            for _ in enumerate(num_blocks):
                layers.append(block(planes))
            return nn.Sequential(*layers)

    def forward(self, inputs):
        """foward function"""
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.PVT_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class MMA_Backbone(nn.Module):
    def __init__(self,in_chans=3, num_stages=4, num_path=[3, 3, 3, 3], num_layers=[3, 4, 6, 3], embed_dims=[64, 128, 192, 256], mlp_ratios=[4, 4, 4, 4], num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1], drop_path_rate=0.0, num_classes=4, **kwargs,):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        # Patch embeddings.
        self.MSP_stages = nn.ModuleList([
            MSP(
                in_chans=embed_dims[idx-1] if idx != 0 else in_chans,
                embed_dim=embed_dims[idx],
                num_path=num_path[idx],
                patch_size = 2 if idx != 0 else 4,
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (PVT_CNN_stages)
        self.PVT_CNN_stages = nn.ModuleList([
            PVT_CNN_stage(
                embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                sr_ratios[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):

        for idx in range(self.num_stages):
            att_inputs = self.MSP_stages[idx](x)
            x = self.PVT_CNN_stages[idx](att_inputs)

        return x


def mma_backbone_tiny(**kwargs):

    model = MMA_Backbone(
        img_size=224,
        num_stages=4,
        num_path=[3, 3, 3, 3],
        num_layers=[2, 2, 2, 2],
        embed_dims=[32, 64, 128, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def mma_backbone_small(**kwargs):

    model = MMA_Backbone(
        img_size=224,
        num_stages=4,
        num_path=[3, 3, 3, 3],
        num_layers=[3, 4, 6, 3],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model

def mma_backbone_medium(**kwargs):

    model = MMA_Backbone(
        img_size=224,
        num_stages=4,
        num_path=[3, 3, 3, 3],
        num_layers=[3, 4, 10, 3],
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


def mma_backbone_large(**kwargs):

    model = MMA_Backbone(
        img_size=224,
        num_stages=4,
        num_path=[3, 3, 3, 3],
        num_layers=[3, 4, 18, 3],
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )

    return model


if __name__ == '__main__':
    x = torch.rand(3, 3, 224, 224)
    model = mma_backbone_small()
    y = model(x)
    print(y.shape)


