import torch
import torch.nn as nn   
import torch.nn.functional as F

class CosyPoseDetectionHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.pose_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 7, kernel_size=1)  # 4 quat + 3 trans
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_map):
        pose_out = self.pose_head(feat_map)  # (B, 7, H, W)
        quat = F.normalize(pose_out[:, 0:4, :, :], dim=1)
        trans = pose_out[:, 4:7, :, :]
        conf = self.conf_head(feat_map).squeeze(1)
        return quat, trans, conf
