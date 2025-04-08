import torch
import torch.nn as nn
import torch.nn.functional as F

class CosyPoseDetectionHead(nn.Module):
    def __init__(self, feature_dim, embed_dim=128):
        super().__init__()
        self.embed_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=1)
        )
        self.pose_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 7, kernel_size=1)
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_map):
        embed = F.normalize(self.embed_head(feat_map), dim=1)
        pose_out = self.pose_head(feat_map)
        quat = F.normalize(pose_out[:, 0:4, :, :], dim=1)
        trans = pose_out[:, 4:7, :, :]
        confidence = self.confidence_head(feat_map)
        return quat, trans, embed, confidence
