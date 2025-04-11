import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from Models.Encoding.FuseNet.FuseDecoder import FuseDecoder
from Models.Encoding.FuseNet.FuseEncoder import FuseEncoder

class FuseNetFeatureEncoder(nn.Module):
    def __init__(self, sensory_channels, ngf=8, out_dim=128, backbone="resnet18"):
        super().__init__()

        self.out_dim = out_dim

        # Step 1: FuseNet encoder-decoder
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf, out_dim=out_dim)

        # Step 2: Project FuseNet output to 3 channels for ResNet
        self.to_rgb_like = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf // 2, 3, kernel_size=1)
        )

        # Step 3: ResNet encoder (no classifier, stops at layer3)
        resnet_fn = getattr(models, backbone)
        base_model = resnet_fn(pretrained=True)
        self.resnet_backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
        )

        # Step 4: Final projection to out_dim (256 → out_dim)
        self.final_proj = nn.Conv2d(256, out_dim, kernel_size=1)

    def forward(self, x_dict):
        # (1) FuseNet fusion
        x_latent, feats = self.encoder(x_dict)
        fused_feat = self.decoder(x_latent, feats)  # (B, out_dim, H, W)

        # (2) Convert to ResNet-compatible input
        rgb_like = self.to_rgb_like(fused_feat)  # (B, 3, H, W)

        # (3) ResNet feature extraction
        res_feat = self.resnet_backbone(rgb_like)  # (B, 256, H/8, W/8)

        # (4) Output projection
        final_feat = self.final_proj(res_feat)  # (B, out_dim, H/8, W/8)

        return final_feat
