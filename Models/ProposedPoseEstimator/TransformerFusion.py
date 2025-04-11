import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads, num_layers):
        super().__init__()
        # Project input channels to transformer embedding dimension.
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.out_dim = hidden_dim
        # Linear layer to project a 3D ray (from intrinsics/extrinsics) into the transformer dimension.
        self.pos_proj = nn.Linear(3, self.out_dim)

    def compute_positional_encoding(self, K, R_w2c, t_w2c, H, W, device):
        """
        Compute normalized 3D rays in world frame (for each pixel) using camera intrinsics and extrinsics.
        K: (3, 3) intrinsics for one view.
        R_w2c: (3, 3) camera rotation (world-to-camera).
        t_w2c: (3,) camera translation (world-to-camera).
        Returns:
            rays_world: (H*W, 3) normalized 3D rays in the world frame.
        """
        # Extract intrinsics.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create a grid of pixel coordinates.
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        # With indexing="xy", grid_u and grid_v will be of shape (H, W)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # grid_u: (H, W), grid_v: (H, W)

        # Convert pixel coordinates into normalized camera coordinates.
        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        z = torch.ones_like(x)
        rays_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

        # Each ray in camera frame is transformed to world frame.
        # Given extrinsics, the camera-to-world rotation is R_c2w = R_w2c^T.
        R_c2w = R_w2c.T
        # Flatten the rays.
        rays_cam_flat = rays_cam.view(-1, 3).T  # (3, H*W)
        rays_world = torch.matmul(R_c2w, rays_cam_flat)  # (3, H*W)
        rays_world = rays_world.T  # (H*W, 3)
        # Normalize the rays.
        rays_world = F.normalize(rays_world, dim=-1)
        return rays_world  # (H*W, 3)

    def forward(self, feature_maps, K_list, extrinsics):
        """
        Args:
            feature_maps: List of tensors, one per view, each of shape (B, C, H, W).
            K_list: List of camera intrinsics (each a tensor of shape (3, 3)), one per view.
            extrinsics: List of dictionaries, one per view, with keys:
                'R_w2c': Tensor of shape (3, 3)
                't_w2c': Tensor of shape (3,)
        Returns:
            fused: Tensor of shape (B, out_dim, H, W)
        """
        B, C, H, W = feature_maps[0].shape
        V = len(feature_maps)
        device = feature_maps[0].device

        # Stack the feature maps along a new view dimension: (B, V, C, H, W)
        stacked = torch.stack(feature_maps, dim=1)
        # Project to hidden dimension per view.
        projected = self.proj(stacked.view(-1, C, H, W))  # (B * V, hidden_dim, H, W)
        projected = projected.view(B, V, self.out_dim, H * W)  # (B, V, hidden_dim, H*W)

        # Compute positional encodings per view.
        pos_encodings = []
        for v in range(V):
            K = K_list[v]
            extrin = extrinsics[v]
            R_w2c = extrin["R_w2c"].to(device)
            t_w2c = extrin["t_w2c"].to(device)
            # Compute 3D rays (H*W, 3) in world frame.
            pe = self.compute_positional_encoding(K, R_w2c, t_w2c, H, W, device)
            pos_encodings.append(pe)
        pos_encodings = torch.stack(pos_encodings, dim=0)  # (V, H*W, 3)
        # Expand to batch dimension: (B, V, H*W, 3)
        pos_encodings = pos_encodings.unsqueeze(0).expand(B, -1, -1, -1)
        # Project positional encoding from 3D to hidden dimension.
        pos_encodings = self.pos_proj(pos_encodings)  # (B, V, H*W, hidden_dim)

        # Rearrange the projected features to match positional encoding shape.
        # From (B, V, hidden_dim, H*W) to (B, H*W, V, hidden_dim)
        tokens = projected.permute(0, 3, 1, 2)

        # Add positional encoding.
        tokens = tokens + pos_encodings  # (B, H*W, V, hidden_dim)

        # Flatten the batch and spatial dimensions: (B * H*W, V, hidden_dim)
        tokens = tokens.flatten(0, 1)

        # Run through the transformer encoder.
        fused_tokens = self.transformer(tokens)  # (B*H*W, V, hidden_dim)
        # Aggregate tokens over the view (V) dimension: here we average.
        fused_tokens = fused_tokens.mean(dim=1)  # (B*H*W, hidden_dim)

        # Reshape back to image space: (B, hidden_dim, H, W)
        fused = fused_tokens.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)
        return fused
