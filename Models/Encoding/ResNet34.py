import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureEncoder(nn.Module):
    def __init__(self, modality="rgb", out_dim=128, backbone="resnet34"):
        super().__init__()

        resnet_fn = getattr(models, backbone)
        base_model = resnet_fn(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modality = modality

        self.feature_extractor = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
        )

        self.out_proj = nn.Conv2d(256, out_dim, kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x_dict):
        x = x_dict[self.modality].unsqueeze(0) # (B, C, H, W)
        feat = self.feature_extractor(x)
        return self.out_proj(feat)  # (B, out_dim, H', W')
