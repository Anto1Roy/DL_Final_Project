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
    def __init__(self, sensory_channels, renderer_config, num_candidates=32, use_learned_similarity=True, ngf=64):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf)
        self.num_candidates = num_candidates
        self.use_learned_similarity = use_learned_similarity

        self.renderer_config = renderer_config
        self.renderer = KaolinRenderer(
            image_size=renderer_config["image_size"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )
        self.render_weight = renderer_config.get("render_weight", 1.0)

        self.sim_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def extract_fused_features(self, x_dict):
        x_latent, encoder_features = self.encoder(x_dict)
        fused_features = self.decoder(x_latent, encoder_features)
        return F.adaptive_avg_pool2d(fused_features, (1, 1)).view(fused_features.size(0), -1)

    def forward(self, x_dict, cad_model_data_list, candidate_pose_list, obj_instance_ids):
        results = []
        for cad_model_data, candidate_poses, instance_ids in zip(
            cad_model_data_list, candidate_pose_list, obj_instance_ids):

            verts, faces = cad_model_data
            
            B, N = candidate_poses.shape[:2]
            device = candidate_poses.device
            
            verts = verts.to(device)
            faces = faces.to(device)

            if verts.dim() == 2:
                verts = verts.unsqueeze(0).repeat(B, 1, 1)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0).repeat(B, 1, 1)

            scene_feat = self.extract_fused_features(x_dict)

            cad_feats = []
            for i in range(N):
                pose = candidate_poses[:, i]
                R = pose[:, :3, :3]
                T = pose[:, :3, 3]

                rendered_imgs = self.renderer(verts, faces, R, T)
                rendered_dict = {"rgb": rendered_imgs.permute(0, 3, 1, 2)}
                rendered_feat = self.extract_fused_features(rendered_dict)
                cad_feats.append(rendered_feat)

            cad_feats = torch.stack(cad_feats, dim=1)
            scene_feat_exp = scene_feat.unsqueeze(1).expand_as(cad_feats)

            if self.use_learned_similarity:
                sim_inputs = torch.cat([scene_feat_exp, cad_feats], dim=-1)
                sim_scores = self.sim_head(sim_inputs).squeeze(-1)
            else:
                sim_scores = F.cosine_similarity(scene_feat_exp, cad_feats, dim=-1)

            best_idx = torch.argmax(sim_scores, dim=1)
            best_pose = torch.stack([candidate_poses[b, idx] for b, idx in enumerate(best_idx)], dim=0)
            R_best = best_pose[:, :3, :3]
            t_best = best_pose[:, :3, 3]

            for i, inst_id in enumerate(instance_ids):
                results.append({
                    "obj_id": int(inst_id),
                    "score": float(sim_scores[i, best_idx[i]].item()),
                    "cam_R_m2c": R_best[i].detach().cpu().numpy().tolist(),
                    "cam_t_m2c": (t_best[i] * 1000.0).detach().cpu().numpy().tolist()
                })

        return results

    def compute_losses(self, x_dict, cad_model_data_list, candidate_pose_list,
                       R_gt_list, t_gt_list, instance_id_list, K=None):
        all_rot_loss, all_trans_loss, all_render_loss = 0.0, 0.0, 0.0
        B_total = 0

        for cad_model_data, candidate_poses, R_gt, t_gt, instance_ids in zip(
                cad_model_data_list, candidate_pose_list, R_gt_list, t_gt_list, instance_id_list):

            preds = self.forward(x_dict, cad_model_data_list, candidate_pose_list, instance_id_list)
            R_pred = torch.tensor([p["cam_R_m2c"] for p in preds], dtype=torch.float32, device=R_gt.device)
            t_pred = torch.tensor([p["cam_t_m2c"] for p in preds], dtype=torch.float32, device=t_gt.device) / 1000.0

            rot_loss = F.mse_loss(R_pred, R_gt)
            trans_loss = F.mse_loss(t_pred, t_gt)

            verts, faces = cad_model_data
            if verts.dim() == 2:
                verts = verts.unsqueeze(0).repeat(R_gt.size(0), 1, 1)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0).repeat(R_gt.size(0), 1, 1)

            mask_pred = self.renderer(verts, faces, R_pred, t_pred, K=K)
            mask_gt = self.renderer(verts, faces, R_gt, t_gt, K=K)
            render_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)

            all_rot_loss += rot_loss.item() * R_gt.size(0)
            all_trans_loss += trans_loss.item() * t_gt.size(0)
            all_render_loss += render_loss.item() * t_gt.size(0)
            B_total += R_gt.size(0)

        avg_rot_loss = all_rot_loss / B_total
        avg_trans_loss = all_trans_loss / B_total
        avg_render_loss = all_render_loss / B_total

        total_loss = avg_rot_loss + avg_trans_loss + self.render_weight * avg_render_loss
        return total_loss, avg_rot_loss, avg_trans_loss, avg_render_loss
