import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.transformer = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.out_dim = hidden_dim

    def compute_positional_encoding(self, K, H, W, device):
        """
        Compute (H, W, 3) normalized ray directions using intrinsics K.
        Output shape: (H*W, 3)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Generate pixel grid
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # (H, W)

        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        z = torch.ones_like(x)

        rays = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
        rays = F.normalize(rays, dim=-1)      # Unit direction

        rays = rays.view(H * W, 3)             # Flattened for token-wise addition
        return rays

    def forward(self, feature_maps, K_list):
        # feature_maps: List of (B, C, H, W)
        # K_list: list of intrinsics matrices, len = num_views, shape = (3, 3)
        B, C, H, W = feature_maps[0].shape
        V = len(feature_maps)
        device = feature_maps[0].device

        # Stack feature maps: (B, V, C, H, W)
        stacked = torch.stack(feature_maps, dim=1)
        projected = self.proj(stacked.view(-1, C, H, W)).view(B, V, -1, H * W)  # (B, V, D, H*W)

        # Prepare positional encodings
        pos_encodings = []
        for v in range(V):
            rays = self.compute_positional_encoding(K_list[v], H, W, device)  # (H*W, 3)
            pos_encodings.append(rays)  # shape (H*W, 3)

        pos_enc = torch.stack(pos_encodings, dim=0).to(device)  # (V, H*W, 3)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1)     # (B, V, H*W, 3)

        # Combine with projected features
        projected = projected.permute(0, 3, 1, 2)  # (B, H*W, V, D)
        pos_enc = pos_enc.permute(0, 2, 1, 3)      # (B, H*W, V, 3)

        # Optionally project positional encodings into same dim as D
        pos_proj = nn.Linear(3, self.out_dim).to(device)
        pe = pos_proj(pos_enc)                    # (B, H*W, V, D)

        tokens = projected + pe                   # (B, H*W, V, D)
        tokens = tokens.flatten(0, 1)             # (B*H*W, V, D)

        fused = self.transformer(tokens)          # (B*H*W, V, D)

        # Average attention-fused output across views
        fused = fused.mean(dim=1).view(B, self.out_dim, H, W)
        return fused
