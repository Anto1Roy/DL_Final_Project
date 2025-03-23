import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiViewTransformerPoseModel(nn.Module):
    def __init__(self, in_channels=3, fc_width=64, feature_dim=1024, num_heads=8, num_layers=4):
        super().__init__()
        ngf = 64
        self.in_channels = in_channels
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        # Encoder (same as FuseNet)
        self.down_1 = Conv_residual_conv(in_channels, ngf, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(ngf, ngf * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(ngf * 2, ngf * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(ngf * 4, ngf * 8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = Conv_residual_conv(ngf * 8, ngf * 16, act_fn)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Transformer encoder for multi-view fusion
        encoder_layer = nn.TransformerEncoderLayer(d_model=ngf * 16, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(ngf * 16, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width // 2), nn.ReLU(),
            nn.Linear(fc_width // 2, 7)
        )

    def forward(self, x, R_gt=None, t_gt=None):
        # x shape: (B, V, C, H, W) where V = number of views
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)

        # Feature extraction
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        bridge = self.bridge(pool_4)
        pooled = self.global_pool(bridge).squeeze(-1).squeeze(-1)  # (B*V, F)

        # Reshape for transformer: (V, B, F)
        pooled = pooled.view(B, V, -1).permute(1, 0, 2)
        fused = self.transformer_encoder(pooled)
        fused = fused.mean(dim=0)  # (B, F)

        pose = self.fc(fused)
        quat = F.normalize(pose[:, :4], dim=1)
        trans = pose[:, 4:]

        if R_gt is not None and t_gt is not None:
            rot_loss = F.mse_loss(quaternion_to_matrix(quat), R_gt)
            trans_loss = F.mse_loss(trans, t_gt)
            return quat, trans, rot_loss, trans_loss
        return quat, trans
