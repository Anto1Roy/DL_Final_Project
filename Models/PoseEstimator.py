import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

# --- Local Imports ---
sys.path.append(os.getcwd())
from Models.helpers import *
from Models.FuseNetModel import FuseEncoder, FuseDecoder
from Models.KaolinRenderer import KaolinRenderer

class CosyPoseStyleRenderMatch(nn.Module):
    def __init__(self, sensory_channels, renderer_config, num_candidates=32, use_learned_similarity=True):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels)
        self.decoder = FuseDecoder(ngf=128)  # Assumes encoder output 128 channels
        self.num_candidates = num_candidates
        self.use_learned_similarity = use_learned_similarity

        self.renderer_config = renderer_config
        self.renderer = KaolinRenderer(
            image_size=renderer_config["image_size"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )

        self.sim_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)  # Scalar similarity score
        )

    def extract_fused_features(self, x_dict):
        x_latent, encoder_features = self.encoder(x_dict)
        fused_features = self.decoder(x_latent, encoder_features)  # (B, C, H, W)
        return F.adaptive_avg_pool2d(fused_features, (1, 1)).view(fused_features.size(0), -1)

    def forward(self, x_dict, cad_model_data, candidate_poses):
        B, N = candidate_poses.shape[:2]
        device = candidate_poses.device

        verts, faces = cad_model_data
        verts = verts.to(device).repeat(B, 1, 1)
        faces = faces.to(device).repeat(B, 1, 1)

        # --- Encode input scene features ---
        scene_feat = self.extract_fused_features(x_dict)  # (B, C)

        cad_feats = []
        for i in range(N):
            pose = candidate_poses[:, i]  # (B, 4, 4)
            R = pose[:, :3, :3]
            T = pose[:, :3, 3]

            rendered_imgs = self.renderer(verts, faces, R, T)  # (B, H, W, 3)
            rendered_dict = {"rgb": rendered_imgs.permute(0, 3, 1, 2)}  # (B, 3, H, W)
            rendered_feat = self.extract_fused_features(rendered_dict)  # (B, C)
            cad_feats.append(rendered_feat)

        cad_feats = torch.stack(cad_feats, dim=1)  # (B, N, C)
        scene_feat_exp = scene_feat.unsqueeze(1).expand_as(cad_feats)  # (B, N, C)

        if self.use_learned_similarity:
            sim_inputs = torch.cat([scene_feat_exp, cad_feats], dim=-1)  # (B, N, 2C)
            sim_scores = self.sim_head(sim_inputs).squeeze(-1)  # (B, N)
        else:
            sim_scores = F.cosine_similarity(scene_feat_exp, cad_feats, dim=-1)  # (B, N)

        best_idx = torch.argmax(sim_scores, dim=1)  # (B,)
        best_pose = torch.stack([candidate_poses[b, idx] for b, idx in enumerate(best_idx)], dim=0)
        R_best = best_pose[:, :3, :3]
        t_best = best_pose[:, :3, 3]

        return R_best, t_best
