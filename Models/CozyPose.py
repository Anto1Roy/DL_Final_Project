import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import *
from Models.KaolinRenderer import KaolinRenderer

class CosyPoseStyleRenderMatch(nn.Module):
    def __init__(self, renderer_config, num_candidates=32, use_learned_similarity=True):
        super().__init__()
        self.num_candidates = num_candidates
        self.use_learned_similarity = use_learned_similarity
        self.render_weight = renderer_config.get("render_weight", 1.0)

        self.renderer = KaolinRenderer(
            width=renderer_config["width"],
            height=renderer_config["height"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )

        self.render_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )

        self.sim_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, feat_map, cad_model_data_list, candidate_pose_list, instance_id_list, K=None):
        results = []
        all_best_R = []
        all_best_t = []

        scene_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(feat_map.size(0), -1)  # (B, F)

        for i in range(len(candidate_pose_list)):
            verts, faces = cad_model_data_list[i]
            candidate_poses = candidate_pose_list[i]
            inst_ids = instance_id_list[i]

            B, N = candidate_poses.shape[:2]
            device = candidate_poses.device

            if verts.dim() == 2:
                verts = verts.unsqueeze(0).repeat(B, 1, 1)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0).repeat(B, 1, 1)

            cad_feats = []
            for n in range(N):
                pose = candidate_poses[:, n]
                R = pose[:, :3, :3]
                T = pose[:, :3, 3]

                rendered = self.renderer(verts, faces, R, T, K=K[i])  # (B, H, W, 3)
                gray = rendered.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # (B, 1, H, W)
                cad_feat = self.render_encoder(gray)
                cad_feats.append(cad_feat)

            cad_feats = torch.stack(cad_feats, dim=1)  # (B, N, F)
            scene_exp = scene_feat[i].unsqueeze(0).expand_as(cad_feats)

            if self.use_learned_similarity:
                sim_input = torch.cat([scene_exp, cad_feats], dim=-1)  # (B, N, 128)
                sim_scores = self.sim_head(sim_input).squeeze(-1)  # (B, N)
            else:
                sim_scores = F.cosine_similarity(scene_exp, cad_feats, dim=-1)

            best_idx = torch.argmax(sim_scores, dim=-1)  # (B,)
            best_pose = torch.stack([candidate_poses[b, idx] for b, idx in enumerate(best_idx)], dim=0)
            all_best_R.append(best_pose[:, :3, :3])
            all_best_t.append(best_pose[:, :3, 3])

            for b in range(B):
                results.append({
                    "obj_id": int(inst_ids[b]),
                    "score": float(sim_scores[b, best_idx[b]].item()),
                    "cam_R_m2c": best_pose[b, :3, :3].detach().cpu().numpy().tolist(),
                    "cam_t_m2c": (best_pose[b, :3, 3] * 1000.0).detach().cpu().numpy().tolist()
                })

        R_tensor = torch.cat(all_best_R, dim=0)
        t_tensor = torch.cat(all_best_t, dim=0)

        return results, R_tensor, t_tensor

    def compute_losses(self, feat_maps, cad_model_data_list, candidate_pose_list,
                       R_gt_list, t_gt_list, instance_id_list, K=None):
        all_rot_loss, all_trans_loss, all_render_loss = 0.0, 0.0, 0.0
        B_total = 0

        for i in range(len(feat_maps)):
            feat_map = feat_maps[i]
            R_gt = torch.stack(R_gt_list[i], dim=0)
            t_gt = torch.stack(t_gt_list[i], dim=0)
            K_i = K[i] if K is not None else None

            _, R_pred, t_pred = self.forward(
                feat_map.unsqueeze(0),
                [cad_model_data_list[i]],
                [candidate_pose_list[i]],
                [instance_id_list[i]],
                K=[K_i] if K_i is not None else None
            )

            rot_loss = F.mse_loss(R_pred, R_gt)
            trans_loss = F.mse_loss(t_pred, t_gt)

            verts, faces = cad_model_data_list[i]
            if verts.dim() == 2:
                verts = verts.unsqueeze(0).repeat(R_gt.size(0), 1, 1)
            if faces.dim() == 2:
                faces = faces.unsqueeze(0).repeat(R_gt.size(0), 1, 1)

            mask_pred = self.renderer(verts, faces, R_pred, t_pred, K=K_i)
            mask_gt = self.renderer(verts, faces, R_gt, t_gt, K=K_i)
            render_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)

            all_rot_loss += rot_loss * R_gt.size(0)
            all_trans_loss += trans_loss * t_gt.size(0)
            all_render_loss += render_loss * t_gt.size(0)
            B_total += R_gt.size(0)

        avg_rot_loss = all_rot_loss / B_total
        avg_trans_loss = all_trans_loss / B_total
        avg_render_loss = all_render_loss / B_total

        total_loss = avg_rot_loss + avg_trans_loss + self.render_weight * avg_render_loss
        return total_loss, avg_rot_loss, avg_trans_loss, avg_render_loss
