import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.KaolinRenderer import KaolinRenderer
from Models.helpers import quaternion_to_matrix, compute_add_s

class SceneRefiner(nn.Module):
    def __init__(self, renderer_config):
        super().__init__()
        self.renderer = KaolinRenderer(
            width=renderer_config["width"],
            height=renderer_config["height"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )

    def compute_loss(self, object_groups, cad_model_lookup, gt_R_tensor=None, gt_t_tensor=None, instance_id_list=None, sample_indices=None, device='cuda'):

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        render_losses = []
        add_s_losses = []

        all_pred_R, all_pred_t, all_gt_R, all_gt_t, all_model_pts = [], [], [], [], []

        for group in object_groups:
            obj_id = group[0].get("instance_id", 0)
            if obj_id not in cad_model_lookup:
                continue
            verts, faces = cad_model_lookup[obj_id]
            verts = verts.to(device)
            faces = faces.to(device)
            verts_batched = verts.unsqueeze(0)
            faces_batched = faces.unsqueeze(0)

            for view in group:
                quat = F.normalize(view['quat'].unsqueeze(0), dim=-1)
                t = view['trans'].unsqueeze(0)
                R = quaternion_to_matrix(quat)
                K = view['K'].unsqueeze(0)
                feat_map = view['feat'].unsqueeze(0)  # (1, C, H, W)

                rendered = self.renderer(verts_batched, faces_batched, R, t, K=K)  # (1, H, W, 3)
                rendered_gray = rendered.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)  # (1,1,H,W)
                rendered_gray = F.interpolate(rendered_gray, size=feat_map.shape[-2:], mode='bilinear', align_corners=False)

                loss = F.mse_loss(rendered_gray, feat_map.mean(dim=1, keepdim=True))
                total_loss = total_loss + loss  # keep it a tensor
                render_losses.append(loss.detach().item())  # just for logging

                if gt_R_tensor is not None and gt_t_tensor is not None and instance_id_list is not None:
                    idx = instance_id_list.index(obj_id)
                    R_gt = gt_R_tensor[idx].unsqueeze(0)
                    t_gt = gt_t_tensor[idx].unsqueeze(0)
                    all_pred_R.append(R)
                    all_pred_t.append(t)
                    all_gt_R.append(R_gt)
                    all_gt_t.append(t_gt)
                    all_model_pts.append(verts.unsqueeze(0))

        # ADD-S loss
        if all_pred_R:
            add_s = compute_add_s(
                torch.cat(all_pred_R),
                torch.cat(all_pred_t),
                torch.cat(all_gt_R),
                torch.cat(all_gt_t),
                torch.cat(all_model_pts)
            )
        else:
            add_s = torch.tensor(0.0, device=device)

        

        # Normalize total loss
        total_loss = total_loss / max(len(render_losses), 1)
        render_losses = torch.tensor(render_losses, device=device).mean()

        return total_loss, render_losses, add_s 

    def forward(self, object_groups, cad_model_lookup):
        device = object_groups[0][0]['quat'].device
        refined_poses = {}
        rendered_views = {}

        for group in object_groups:
            obj_id = group[0].get("instance_id", 0)
            if obj_id not in cad_model_lookup:
                continue
            verts, faces = cad_model_lookup[obj_id]
            verts = verts.to(device).unsqueeze(0)
            faces = faces.to(device).unsqueeze(0)

            refined_group = []
            rendered_group = []

            for view in group:
                quat = F.normalize(view['quat'], dim=-1)
                trans = view['trans']
                pose = torch.eye(4, device=device)
                pose[:3, :3] = quaternion_to_matrix(quat.unsqueeze(0))[0]
                pose[:3, 3] = trans

                refined_group.append({
                    "quat": quat,
                    "trans": trans,
                    "pose_matrix": pose,
                    "score": view.get("score", 1.0)
                })

                R = quaternion_to_matrix(quat.unsqueeze(0))
                T = trans.unsqueeze(0)
                K = view['K'].unsqueeze(0)
                rendered = self.renderer(verts, faces, R, T, K=K)  # (1, H, W, 3)
                rendered_group.append(rendered)

            refined_poses[obj_id] = refined_group
            rendered_views[obj_id] = rendered_group

        return refined_poses, rendered_views