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

    def forward(self, feat_maps, cad_models, candidate_poses_list, instance_ids, K_list=None):
        results = []
        all_best_R = []
        all_best_t = []
        for i in range(len(cad_models)):
            feat_map = F.adaptive_avg_pool2d(feat_maps[i], (1, 1)).view(1, -1)  # (1, F)
            verts, faces = cad_models[i]
            candidate_poses = candidate_poses_list[i]  # (N, 4, 4)
            obj_id = instance_ids[i]
            K = K_list[i] if K_list is not None else None

            B, N = 1, candidate_poses.shape[0]
            device = candidate_poses.device

            verts = verts.unsqueeze(0) if verts.dim() == 2 else verts
            faces = faces.unsqueeze(0) if faces.dim() == 2 else faces

            cad_feats = []
            for n in range(N):
                pose = candidate_poses[n].unsqueeze(0)  # (1, 4, 4)
                R = pose[:, :3, :3]
                T = pose[:, :3, 3]

                rendered = self.renderer(verts, faces, R, T, K=K)  # (1, H, W, 3)
                gray = rendered.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # (1, 1, H, W)
                cad_feat = self.render_encoder(gray)  # (1, F)
                cad_feats.append(cad_feat)

            cad_feats = torch.cat(cad_feats, dim=0).unsqueeze(0)  # (1, N, F)
            scene_exp = feat_map.unsqueeze(1).expand_as(cad_feats)  # (1, N, F)

            if self.use_learned_similarity:
                sim_input = torch.cat([scene_exp, cad_feats], dim=-1)  # (1, N, 128)
                sim_scores = self.sim_head(sim_input).squeeze(0).squeeze(-1)  # (N,)
            else:
                sim_scores = F.cosine_similarity(scene_exp.squeeze(0), cad_feats.squeeze(0), dim=-1)  # (N,)

            best_idx = torch.argmax(sim_scores, dim=0)  # ()
            best_pose = candidate_poses[best_idx]  # (4, 4)

            all_best_R.append(best_pose[:3, :3].unsqueeze(0))
            all_best_t.append(best_pose[:3, 3].unsqueeze(0))

            results.append({
                "obj_id": int(obj_id),
                "score": float(sim_scores[best_idx].item()),
                "cam_R_m2c": best_pose[:3, :3].detach().cpu().numpy().tolist(),
                "cam_t_m2c": (best_pose[:3, 3] * 1000.0).detach().cpu().numpy().tolist()
            })

        R_tensor = torch.cat(all_best_R, dim=0)
        t_tensor = torch.cat(all_best_t, dim=0)

        return results, R_tensor, t_tensor


    def compute_losses(self, feat_maps, cad_model_data_list, candidate_pose_list,
                   R_gt_list, t_gt_list, instance_id_list, K_list=None):
        device = feat_maps[0].device 
        R_gt_tensor = torch.stack(R_gt_list).to(device)
        t_gt_tensor = torch.stack(t_gt_list).to(device)

        # Forward pass to get the predicted poses
        _, R_pred, t_pred = self.forward(
            feat_maps,
            cad_model_data_list,
            candidate_pose_list,
            instance_id_list,
            K_list=K_list
        )

        # Rotation and Translation Losses
        rot_loss = F.mse_loss(R_pred, R_gt_tensor)
        trans_loss = F.mse_loss(t_pred, t_gt_tensor)

        # Render losses for each sample
        render_losses = []
        for i in range(len(R_gt_list)):
            verts, faces = cad_model_data_list[i]

            verts = verts.unsqueeze(0)  # (1, V, 3)
            faces = faces.unsqueeze(0)  # (1, F, 3)

            # Extract predicted and ground truth poses for each object
            R_pred_i = R_pred[i].unsqueeze(0)
            t_pred_i = t_pred[i].unsqueeze(0)
            R_gt_i = R_gt_tensor[i].unsqueeze(0)
            t_gt_i = t_gt_tensor[i].unsqueeze(0)
            K_i = K_list[i].unsqueeze(0) if K_list is not None else None

            # Render the predicted and ground truth masks
            mask_pred = self.renderer(verts, faces, R_pred_i, t_pred_i, K=K_i)
            mask_gt = self.renderer(verts, faces, R_gt_i, t_gt_i, K=K_i)

            # Compute binary cross-entropy loss between predicted and ground truth masks
            render_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)
            render_losses.append(render_loss)

        # Average render loss
        avg_render_loss = torch.stack(render_losses).mean()

        # If render loss is NaN or Inf, skip rendering loss and calculate only the rotation and translation loss
        if torch.isnan(avg_render_loss) or torch.isinf(avg_render_loss):
            total_loss = rot_loss + trans_loss
        else:
            # Total loss includes rotation, translation, and weighted render loss
            total_loss = rot_loss + trans_loss + self.render_weight * avg_render_loss

        return total_loss, rot_loss, trans_loss, avg_render_loss