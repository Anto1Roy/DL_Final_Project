import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureEncoder(nn.Module):
    def __init__(self, modality="rgb", out_dim=64, backbone="resnet18"):
        super().__init__()

        resnet_fn = getattr(models, backbone)
        base_model = resnet_fn(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.modality = modality

        # Extract all layers up to conv5
        self.feature_extractor = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        # Reduce channels to out_dim (e.g., 64)
        self.out_proj = nn.Conv2d(base_model.fc.in_features, out_dim, kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x_dict):
        outputs = []
        for view_idx in range(len(x_dict[self.modality])):  # Iterate over views
            x = x_dict[self.modality][view_idx]  # (B, C, H, W)
            feat = self.feature_extractor(x)
            projected_feat = self.out_proj(feat)  # (B, out_dim, H', W')
            outputs.append(projected_feat)
        return outputs  # List of tensors, one per view
