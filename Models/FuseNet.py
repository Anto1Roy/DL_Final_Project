import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# --- Local Import ---
sys.path.append(os.getcwd())
from Models.helpers import *
from Models.FuseEncoder import FuseEncoder
from Models.FuseDecoder import FuseDecoder

class FuseNetPoseModel(nn.Module):
    def __init__(self, sensory_channels, fc_width=64):
        super().__init__()
        ngf = 64
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(ngf, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width), nn.ReLU(),
            nn.Linear(fc_width, fc_width // 2), nn.ReLU(),
            nn.Linear(fc_width // 2, 7)
        )

    def forward(self, x_dict, R_gt=None, t_gt=None):
        x, feats = self.encoder(x_dict)
        x = self.decoder(x, feats)

        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        pose = self.fc(pooled)

        quat = F.normalize(pose[:, :4], dim=1)
        trans = pose[:, 4:]

        if R_gt is not None and t_gt is not None:
            rot_loss = F.mse_loss(quaternion_to_matrix(quat), R_gt)
            trans_loss = F.mse_loss(trans, t_gt)
            return quat, trans, rot_loss, trans_loss
        return quat, trans
