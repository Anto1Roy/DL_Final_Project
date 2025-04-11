import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from Models.KaolinRenderer import KaolinRenderer
from Models.helpers import quaternion_to_matrix

def transform_points(R, t, pts):
    return R @ pts.T + t.view(3, 1)  # [3, N]

class SceneRefiner(nn.Module):
    def __init__(self, renderer_config):
        super().__init__()
        self.renderer = KaolinRenderer(
            width=renderer_config["width"],
            height=renderer_config["height"],
            fov=renderer_config.get("fov", 60.0),
            device=renderer_config["device"]
        )
        self.device = renderer_config["device"]

    def refine_scene(self, detections, gt_views, cad_model_lookup, num_iters=100, lr=1e-2):
        """
        Jointly refine object and camera poses.
        detections: list of merged global predictions (same object merged across views)
        gt_views: list of per-view GTs with intrinsics and camera extrinsics
        cad_model_lookup: dict[obj_id] -> (verts, faces)
        """
        device = self.device

        # --- Step 1: Create optimization parameters ---
        obj_params = {}
        for det in detections:
            instance_id = det['instance_id']
            if instance_id not in obj_params:
                q = det['quat'].detach().clone().requires_grad_()
                t = det['trans'].detach().clone().requires_grad_()
                obj_params[instance_id] = {'quat': q, 'trans': t}

        cam_params = {}
        for view in gt_views:
            view_id = view['view_id']
            if view_id not in cam_params:
                R = view['cam_extrinsics'][:3, :3].detach().clone()
                t = view['cam_extrinsics'][:3, 3].detach().clone()
                R = R.requires_grad_()
                t = t.requires_grad_()
                cam_params[view_id] = {'R': R, 't': t}

        optimizer = Adam(
            list([p for obj in obj_params.values() for p in obj.values()]) +
            list([p for cam in cam_params.values() for p in cam.values()]), lr=lr)

        # --- Step 2: Optimization loop ---
        for _ in range(num_iters):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            
            for view in gt_views:
                K = view['K'].unsqueeze(0)
                view_id = view['view_id']
                cam_ext = torch.eye(4, device=device)
                cam_ext[:3, :3] = cam_params[view_id]['R']
                cam_ext[:3, 3] = cam_params[view_id]['t']

                for det in detections:
                    if det['instance_id'] != view['instance_id']:
                        continue

                    instance_id = det['instance_id']
                    if instance_id not in cad_model_lookup:
                        continue

                    verts, faces = cad_model_lookup[instance_id]
                    verts = verts.to(device).unsqueeze(0)
                    faces = faces.to(device).unsqueeze(0)

                    R_obj = quaternion_to_matrix(obj_params[instance_id]['quat'].unsqueeze(0))
                    t_obj = obj_params[instance_id]['trans'].unsqueeze(0)

                    rendered = self.renderer(verts, faces, R_obj, t_obj, K=K, cam_extrinsics=cam_ext.unsqueeze(0))
                    rendered_gray = rendered.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)

                    # Use GT render or silhouette loss if available, here dummy target is used
                    target = view['rendered_gt'] if 'rendered_gt' in view else torch.zeros_like(rendered_gray)
                    loss = F.mse_loss(rendered_gray, target)
                    total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

        # Return refined poses
        refined = []
        for instance_id, pose in obj_params.items():
            refined.append({
                'instance_id': instance_id,
                'quat': F.normalize(pose['quat'].detach(), dim=0),
                'trans': pose['trans'].detach()
            })
        return refined
