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
        self.pools = nn.ModuleList([maxpool() for _ in range(4)])

        self.bridge = Conv_residual_conv(ngf * 8, ngf * 16, self.act_fn)

    def forward(self, x_dict):
        feats = []
        mod_feats = {}
        x_fused = None

        for i in range(4):
            level_feats = []
            for mod in self.modalities:
                if i == 0:
                    f = self.encoders[mod][i](x_dict[mod].unsqueeze(0))
                else:
                    f = self.encoders[mod][i](self.pools[i - 1](mod_feats[mod]))
                level_feats.append(f)
                mod_feats[mod] = f

            # use sum because concat fusion is too heavy
            x_fused = sum(level_feats) / len(level_feats)
            feats.append(x_fused)

        x = self.pools[3](x_fused)
        x = self.bridge(x)
        return x, feats
