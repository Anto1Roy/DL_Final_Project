import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ObjectDetection.FuseNet.FuseEncoder import FuseEncoder
from Models.ObjectDetection.FuseNet.FuseDecoder import FuseDecoder

class FuseNetFeatureEncoder(nn.Module):
    def __init__(self, sensory_channels, ngf=64):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf)
        self.out_dim = ngf  # decoder output channels

    def forward(self, x_dict):
        outputs = []
        for view_idx in range(len(next(iter(x_dict.values())))):  # Iterate over views
            x_dict_view = {modality: x_dict[modality][view_idx] for modality in x_dict}
            x_latent, feats = self.encoder(x_dict_view)
            decoded = self.decoder(x_latent, feats)  # (B, ngf, H, W)
            outputs.append(decoded)
        return outputs  # List of tensors, one per view
