import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewTransformer(nn.Module):
    def __init__(self, num_views, feature_dim=256):
        super(MultiViewTransformer, self).__init__()
        self.num_views = num_views
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=2
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7)  # 4 for quaternion, 3 for translation
        )

    def forward(self, x):
        # x: (B, num_views, 1, H, W)
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)  # (B*V, 1, H, W)
        features = self.cnn_encoder(x).squeeze(-1).squeeze(-1)  # (B*V, 128)
        features = features.view(B, V, -1).permute(1, 0, 2)  # (V, B, 128)
        fused_features = self.transformer_encoder(features).mean(dim=0)  # (B, 128)
        pose = self.fc(fused_features)
        quat = F.normalize(pose[:, :4], dim=1)
        trans = pose[:, 4:]
        return quat, trans
