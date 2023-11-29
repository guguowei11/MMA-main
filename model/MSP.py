from torch import nn
import torch

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,kernel_size, patch_size=1, in_chans=3, embed_dim=768, padding=1):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.Relu = nn.ReLU(True)

    def forward(self, x):

        x = self.proj(x)
        x = self.norm(x)
        x = self.Relu(x)

        return x


class MSP(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `PatchEmbed` layers."""
    def __init__(self,in_chans, embed_dim, patch_size, num_path=3, kernel_size=[3,5,7],kernel_size2=[5,7,11]):
        super(MSP, self).__init__()

        self.patch_embeds = nn.ModuleList([
            PatchEmbed(
                in_chans= in_chans,
                embed_dim= embed_dim,
                kernel_size= kernel_size[idx] if in_chans != 3 else kernel_size2[idx],
                patch_size = patch_size,
                padding = kernel_size[idx]//2 if in_chans != 3 else kernel_size2[idx]//2,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """foward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            h = pe(x)
            att_inputs.append(h)

        return att_inputs

if __name__ == '__main__':
    x = torch.rand(3, 3, 224, 224)
    model = MSP(3,3,2)
    y = model(x)
    print(y[0].shape)