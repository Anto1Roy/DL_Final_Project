# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# --- Local Import ---
sys.path.append(os.getcwd())
from Models.helpers import *


class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.conv_1 = conv_block(in_dim, out_dim, act_fn)
        self.conv_2 = conv_block_3(out_dim, out_dim, act_fn)
        self.conv_3 = conv_block(out_dim, out_dim, act_fn)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FuseNetPoseModel(nn.Module):
    def __init__(self, sensory_channels, fc_width=64):
        super().__init__()
        ngf = 64
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating Dual-Encoder FuseNetPoseModel------\n")

        # Create modality-specific encoders
        self.modalities = list(sensory_channels.keys())
        self.encoders = nn.ModuleDict()

        for mod in self.modalities:
            in_ch = sensory_channels[mod]
            self.encoders[mod] = nn.ModuleList([
                Conv_residual_conv(in_ch, ngf, act_fn),
                Conv_residual_conv(ngf, ngf * 2, act_fn),
                Conv_residual_conv(ngf * 2, ngf * 4, act_fn),
                Conv_residual_conv(ngf * 4, ngf * 8, act_fn),
                Conv_residual_conv(ngf * 8, ngf * 16, act_fn)
            ])

        self.pools = nn.ModuleList([maxpool() for _ in range(5)])

        # Bridge
        self.bridge = Conv_residual_conv(ngf * 16, ngf * 32, act_fn)

        # Decoder
        self.deconv = nn.ModuleList([
            conv_trans_block(ngf * 32, ngf * 16, act_fn_2),
            conv_trans_block(ngf * 16, ngf * 8, act_fn_2),
            conv_trans_block(ngf * 8, ngf * 4, act_fn_2),
            conv_trans_block(ngf * 4, ngf * 2, act_fn_2),
            conv_trans_block(ngf * 2, ngf, act_fn_2),
        ])

        self.up = nn.ModuleList([
            Conv_residual_conv(ngf * 16, ngf * 16, act_fn_2),
            Conv_residual_conv(ngf * 8, ngf * 8, act_fn_2),
            Conv_residual_conv(ngf * 4, ngf * 4, act_fn_2),
            Conv_residual_conv(ngf * 2, ngf * 2, act_fn_2),
            Conv_residual_conv(ngf, ngf, act_fn_2),
        ])

        self.refine = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
        )

        # Pose Regression Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(ngf, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width // 2), nn.ReLU(),
            nn.Linear(fc_width // 2, 7)
        )

    def forward(self, x_dict, R_gt=None, t_gt=None):
        assert set(x_dict.keys()) == set(self.modalities)

        feats = []  # fused features per encoder stage
        x_fused = None
        mod_feats = {}

        for i in range(5):
            level_feats = []
            for mod in self.modalities:
                if i == 0:
                    f = self.encoders[mod][i](x_dict[mod])
                else:
                    f = self.encoders[mod][i](self.pools[i - 1](mod_feats[mod]))
                level_feats.append(f)
                mod_feats[mod] = f

            x_fused = sum(level_feats) / len(level_feats)
            feats.append(x_fused)
            mod_feats = {mod: f for mod, f in zip(self.modalities, level_feats)}

        x = self.pools[4](x_fused)
        x = self.bridge(x)

        for i in range(5):
            x = self.deconv[i](x)
            x = (x + feats[4 - i]) / 2
            x = self.up[i](x)

        x = self.refine(x)
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        pose = self.fc(pooled)

        quat = F.normalize(pose[:, :4], dim=1)
        trans = pose[:, 4:]

        if R_gt is not None and t_gt is not None:
            rot_loss = F.mse_loss(quaternion_to_matrix(quat), R_gt)
            trans_loss = F.mse_loss(trans, t_gt)
            return quat, trans, rot_loss, trans_loss
        return quat, trans
