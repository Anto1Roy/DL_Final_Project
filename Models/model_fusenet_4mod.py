import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseNet(nn.Module):
    def __init__(self, in_channels):
        super(FuseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 7)  # 4 for quaternion, 3 for translation
        )

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)  # (B, 256)
        pose = self.fc(x)
        quat = F.normalize(pose[:, :4], dim=1)  # Normalize quaternion
        trans = pose[:, 4:]
        return quat, trans
