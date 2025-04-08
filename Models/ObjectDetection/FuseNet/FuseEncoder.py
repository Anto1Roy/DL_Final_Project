import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# --- Local Import ---
sys.path.append(os.getcwd())
from Models.helpers import *

class FuseEncoder(nn.Module):
    def __init__(self, sensory_channels, ngf=64, act_fn=None):
        super().__init__()
        self.modalities = list(sensory_channels.keys())
        self.ngf = ngf
        self.act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)

        # Modality-specific encoders
        self.encoders = nn.ModuleDict()
        for mod in self.modalities:
            in_ch = sensory_channels[mod]
            self.encoders[mod] = nn.ModuleList([
                Conv_residual_conv(in_ch, ngf, self.act_fn),
                Conv_residual_conv(ngf, ngf * 2, self.act_fn),
                Conv_residual_conv(ngf * 2, ngf * 4, self.act_fn),
                Conv_residual_conv(ngf * 4, ngf * 8, self.act_fn),
            ])
        self.pools = nn.ModuleList([maxpool() for _ in range(5)])

        # Fusion layers (learnable 1x1 convs after concat)
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(len(self.modalities) * ngf, ngf, kernel_size=1),
            nn.Conv2d(len(self.modalities) * ngf * 2, ngf * 2, kernel_size=1),
            nn.Conv2d(len(self.modalities) * ngf * 4, ngf * 4, kernel_size=1),
            nn.Conv2d(len(self.modalities) * ngf * 8, ngf * 8, kernel_size=1),
        ])

        self.bridge = Conv_residual_conv(ngf * 8, ngf * 16, self.act_fn)

    def forward(self, x_dict):
        feats = []
        mod_feats = {}
        x_fused = None

        for i in range(4):
            level_feats = []
            for mod in self.modalities:
                if i == 0:
                    f = self.encoders[mod][i](x_dict[mod])
                else:
                    f = self.encoders[mod][i](self.pools[i - 1](mod_feats[mod]))
                level_feats.append(f)
                mod_feats[mod] = f

            # Concatenate along channel dim
            x_cat = torch.cat(level_feats, dim=1)  # [B, num_modalities * C, H, W]
            x_fused = self.fusion_convs[i](x_cat)  # [B, C, H, W]
            feats.append(x_fused)

        x = self.pools[4](x_fused)
        x = self.bridge(x)
        return x, feats
