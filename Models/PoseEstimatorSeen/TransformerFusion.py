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
        self.pos_proj = nn.Linear(3, self.out_dim)  # moved to init

    def compute_positional_encoding(self, K, R_w2c, t_w2c, H, W, device):
        """
        Compute normalized 3D rays in world frame for each pixel using intrinsics and extrinsics.
        Output shape: (H*W, 3)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # (H, W)

        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        z = torch.ones_like(x)
        rays_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

        # Convert rays to world frame: X_w = R^T * (X_c - t)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c.view(3, 1)

        rays_cam_flat = rays_cam.view(-1, 3).T  # (3, N)
        rays_world = R_c2w @ rays_cam_flat  # (3, N)
        rays_world = F.normalize(rays_world.T, dim=-1)  # (N, 3)

        return rays_world  # (H*W, 3)

    def forward(self, feature_maps, K_list, extrinsics):
        B, C, H, W = feature_maps[0].shape
        V = len(feature_maps)
        device = feature_maps[0].device

        stacked = torch.stack(feature_maps, dim=1)  # (B, V, C, H, W)
        projected = self.proj(stacked.view(-1, C, H, W)).view(B, V, -1, H * W)  # (B, V, D, H*W)

        pos_encodings = []
        for v in range(V):
            K = K_list[v]
            R = extrinsics[v]["R_w2c"].to(device)
            t = extrinsics[v]["t_w2c"].to(device)
            rays = self.compute_positional_encoding(K, R, t, H, W, device)  # (H*W, 3)
            pos_encodings.append(rays)

        pos_enc = torch.stack(pos_encodings, dim=0)  # (V, H*W, 3)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1)  # (B, V, H*W, 3)

        projected = projected.permute(0, 3, 1, 2)  # (B, H*W, V, D)
        pos_enc = pos_enc.permute(0, 2, 1, 3)      # (B, H*W, V, 3)

        pe = self.pos_proj(pos_enc)               # (B, H*W, V, D)
        tokens = projected + pe                   # (B, H*W, V, D)
        tokens = tokens.flatten(0, 1)             # (B*H*W, V, D)

        fused = self.transformer(tokens)          # (B*H*W, V, D)
        fused = fused.mean(dim=1).view(B, self.out_dim, H, W)
        return fused