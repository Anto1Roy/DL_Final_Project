# --- Imports ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

# from kaolin.render.camera import generate_perspective_projection_matrix
from kaolin.render.mesh import rasterize
# from kaolin.ops.mesh import index_vertices_by_faces


class KaolinRenderer(nn.Module):
    def __init__(self, image_size=256, fov=60.0, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.fov = fov
        self.device = device

    def forward(self, verts, faces, R, T):
        B = R.shape[0]
        images = []

        proj = self.get_projection_matrix(self.fov, near=0.01, far=10.0).to(self.device)

        for b in range(B):
            cam_rot = R[b]
            cam_pos = T[b]

            world2cam = torch.eye(4, device=self.device)
            world2cam[:3, :3] = cam_rot.T
            world2cam[:3, 3] = -cam_rot.T @ cam_pos

            VP = proj @ world2cam  # (4, 4)

            V = verts.shape[1]
            ones = torch.ones((V, 1), device=verts.device, dtype=verts.dtype)
            verts_b = verts[b]  # (V, 3)
            verts_homo = torch.cat([verts_b, ones], dim=-1)  # (V, 4)

            verts_cam = verts_homo @ VP.T  # (V, 4)
            verts_ndc = verts_cam[:, :3] / verts_cam[:, 3:].clamp(min=1e-8)  # (V, 3)

            face_b = faces[b]  # (F, 3)
            face_features = torch.ones((face_b.shape[0], 3), device=verts.device)  # (F, 3)

            rast_out = rasterize(
                verts_ndc[b],        # (1, V, 3)
                face_b[b],           # (1, F, 3)
                face_features[b],    # (1, F, 3)
                self.image_size,
                self.image_size
            )

            mask = rast_out['mask'][0].float().unsqueeze(-1).repeat(1, 1, 3)  # (H, W, 3)
            images.append(mask)

        return torch.stack(images, dim=0)  # (B, H, W, 3)

    
    def get_projection_matrix(self, fov_deg, near, far):
        fov_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_rad / 2)
        proj = torch.tensor([
            [f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)
        return proj