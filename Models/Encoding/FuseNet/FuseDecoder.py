import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# --- Local Import ---
sys.path.append(os.getcwd())
from Models.helpers import *

class FuseDecoder(nn.Module):
    def __init__(self, ngf=64, out_dim=128, act_fn=None):
        super().__init__()
        act_fn = act_fn or nn.ReLU()
        self.deconv = nn.ModuleList([
            conv_trans_block(ngf * 16, ngf * 8, act_fn),
            conv_trans_block(ngf * 8, ngf * 4, act_fn),
            conv_trans_block(ngf * 4, ngf * 2, act_fn),
            conv_trans_block(ngf * 2, ngf, act_fn),
        ])
        self.up = nn.ModuleList([
            Conv_residual_conv(ngf * 8, ngf * 8, act_fn),
            Conv_residual_conv(ngf * 4, ngf * 4, act_fn),
            Conv_residual_conv(ngf * 2, ngf * 2, act_fn),
            Conv_residual_conv(ngf, ngf, act_fn),
        ])
        # self.refine = nn.Sequential(
        #     nn.Conv2d(ngf, out_dim, kernel_size=3, padding=1),   # make sure it outputs out_dim
        #     nn.BatchNorm2d(out_dim),
        #     nn.ReLU(),
        # )

    def forward(self, x, feats):
        for i in range(4):
            x = self.deconv[i](x)
            x = (x + feats[3 - i]) / 2
            x = self.up[i](x)
        # x = self.refine(x)
        return x
