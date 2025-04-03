import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

# --- Local Imports ---
sys.path.append(os.getcwd())
from Models.helpers import *
from Models.FuseNetModel import FuseEncoder
from Models.KaolinRenderer import KaolinRenderer

class RenderMatchPoseNet(nn.Module):
    def __init__(self, sensory_channels, renderer_config, num_candidates=32):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels)
        self.num_candidates = num_candidates

        self.renderer_config = renderer_config
        self.renderer = KaolinRenderer(
            image_size=renderer_config["image_size"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )

        self.pose_fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_dict, cad_model_data, candidate_poses):
        """
        Args:
            x_dict: dict of modality → (B, C, H, W)
            cad_model_data: tuple of (vertices, faces) from Kaolin
            candidate_poses: (B, N, 4, 4)
        Returns:
            R_best: (B, 3, 3), t_best: (B, 3)
        """
        B, N = candidate_poses.shape[:2]
        device = candidate_poses.device

        verts, faces = cad_model_data  # unpack mesh

        print(verts.shape, faces.shape)
        # verts: (V, 3), faces: (F, 3)

        # Repeat verts and faces for batch size B
        verts = verts.to(device).repeat(B, 1, 1)   # (B, V, 3)
        faces = faces.to(device).repeat(B, 1, 1)   # (B, F, 3)

        # Encode input scene
        x_latent, _ = self.encoder(x_dict)
        x_feat = F.adaptive_avg_pool2d(x_latent, (1, 1)).view(B, -1)

        cad_feats = []
        for i in range(N):
            pose = candidate_poses[:, i]  # (B, 4, 4)
            R = pose[:, :3, :3]
            T = pose[:, :3, 3]

            images = self.renderer(verts, faces, R, T)  # (B, H, W, 3)

            rendered_dict = {"rgb": images.permute(0, 3, 1, 2)}
            cad_latent, _ = self.encoder(rendered_dict)
            cad_feat = F.adaptive_avg_pool2d(cad_latent, (1, 1)).view(B, -1)
            cad_feats.append(cad_feat)

        cad_feats = torch.stack(cad_feats, dim=1)  # (B, N, C)

        x_feat_exp = x_feat.unsqueeze(1).expand_as(cad_feats)
        sim = F.cosine_similarity(x_feat_exp, cad_feats, dim=-1)
        best_idx = torch.argmax(sim, dim=1)

        best_pose = torch.stack([candidate_poses[b, idx] for b, idx in enumerate(best_idx)], dim=0)
        R_best = best_pose[:, :3, :3]
        t_best = best_pose[:, :3, 3]

        return R_best, t_best
