import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.out_dim = hidden_dim

    def forward(self, feature_maps):
        # feature_maps: List of (B, C, H, W)
        B, C, H, W = feature_maps[0].shape
        V = len(feature_maps)

        # Stack and flatten spatial dims: (V, B, C, H, W) -> (B, V, C, H*W)
        stacked = torch.stack(feature_maps, dim=1)
        projected = self.proj(stacked.view(-1, C, H, W)).view(B, V, -1, H * W)
        
        # Transpose to: (B, H*W, V, D)
        tokens = projected.permute(0, 3, 1, 2)  # B x H*W x V x D

        # Merge batch and spatial dims: (B * H*W, V, D)
        tokens = tokens.flatten(0, 1)

        # Run transformer
        fused = self.transformer(tokens)  # Same shape

        # Reshape back to (B, D, H, W)
        fused = fused[:, 0, :].view(B, -1, H, W)  # Take only first token or use avg pool
        return fused
