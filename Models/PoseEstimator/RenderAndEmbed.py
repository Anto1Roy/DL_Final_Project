import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import resize, normalize

class RenderAndEmbed(nn.Module):
    def __init__(self, renderer, output_dim=128, input_size=(224, 224)):
        super().__init__()
        self.renderer = renderer
        self.input_size = input_size

        # Use pretrained ResNet18 from torchvision
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()  # Remove classifier head
        self.backbone = resnet

        # Projection head to custom output_dim
        self.projection = nn.Linear(512, output_dim)

    def forward(self, verts, faces, R, T, K=None):
        # (1, H, W, 3) → (1, 3, H, W)
        render = self.renderer(
            verts.unsqueeze(0), faces.unsqueeze(0),
            R.unsqueeze(0), T.unsqueeze(0), K=K
        )
        render = render.permute(0, 3, 1, 2)  # to (B, C, H, W)

        # Resize and normalize for ResNet
        render = F.interpolate(render, size=self.input_size, mode="bilinear", align_corners=False)

        render = render.expand(-1, 3, -1, -1)

        render = normalize(render, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Pass through CNN
        features = self.backbone(render)
        proj = self.projection(features)
        return F.normalize(proj, dim=1).squeeze(0)  # (D,)
