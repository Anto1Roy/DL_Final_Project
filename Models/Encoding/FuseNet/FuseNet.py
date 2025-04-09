import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Encoding.FuseNet.FuseEncoder import FuseEncoder
from Models.Encoding.FuseNet.FuseDecoder import FuseDecoder

class FuseNetFeatureEncoder(nn.Module):
    def __init__(self, sensory_channels, ngf=16, out_dim=128):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf, out_dim=out_dim)
        self.out_dim = out_dim  # decoder output channels

    def forward(self, x_dict):
        x_latent, feats = self.encoder(x_dict)
        decoded = self.decoder(x_latent, feats)  # (B, ngf, H, W)
        return decoded
