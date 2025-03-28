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
    def __init__(self, in_channels=2, fc_width=128):
        super().__init__()
        ngf = 128
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FuseNetPoseModel------\n")

        # Encoder
        self.down_1 = Conv_residual_conv(in_channels, ngf, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(ngf, ngf * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(ngf * 2, ngf * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(ngf * 4, ngf * 8, act_fn)
        self.pool_4 = maxpool()
        self.down_5 = Conv_residual_conv(ngf * 8, ngf * 16, act_fn)
        self.pool_5 = maxpool()

        # Bridge
        self.bridge = Conv_residual_conv(ngf * 16, ngf * 32, act_fn)

        # Decoder
        self.deconv_0 = conv_trans_block(ngf * 32, ngf * 16, act_fn_2)
        self.up_0 = Conv_residual_conv(ngf * 16, ngf * 16, act_fn_2)
        self.deconv_1 = conv_trans_block(ngf * 16, ngf * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(ngf * 8, ngf * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(ngf * 8, ngf * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(ngf * 4, ngf * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(ngf * 4, ngf * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(ngf * 2, ngf * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(ngf * 2, ngf, act_fn_2)
        self.up_4 = Conv_residual_conv(ngf, ngf, act_fn_2)

        # Refinement block
        self.refine = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
        )

        # Pose Regression Head (multi-layered)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(ngf, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width // 2), nn.ReLU(),
            nn.Linear(fc_width // 2, 7)
        )

    def forward(self, x, R_gt=None, t_gt=None):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        bridge = self.bridge(pool_5)

        deconv_0 = self.deconv_0(bridge)
        skip_0 = (deconv_0 + down_5) / 2
        up_0 = self.up_0(skip_0)

        deconv_1 = self.deconv_1(up_0)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)

        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)

        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)

        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        refined = self.refine(up_4)
        pooled = self.global_pool(refined).squeeze(-1).squeeze(-1)
        pose = self.fc(pooled)
        quat = F.normalize(pose[:, :4], dim=1)
        trans = pose[:, 4:]

        if R_gt is not None and t_gt is not None:
            rot_loss = F.mse_loss(quaternion_to_matrix(quat), R_gt)
            trans_loss = F.mse_loss(trans, t_gt)
            return quat, trans, rot_loss, trans_loss
        return quat, trans
