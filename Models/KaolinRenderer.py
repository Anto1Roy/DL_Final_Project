# --- Imports ---
import math
import torch
import torch.nn as nn

from kaolin.render.mesh import rasterize


class KaolinRenderer(nn.Module):
    def __init__(self, image_size=256, fov=60.0, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.fov = fov
        self.device = device

    def forward(self, verts, faces, R, T, K=None):
        B = R.shape[0]
        images = []

        # Ensure verts and faces are batched
        if verts.dim() == 2:
            verts = verts.unsqueeze(0).repeat(B, 1, 1)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).repeat(B, 1, 1)

        print(verts.shape, faces.shape, R.shape, T.shape)

        for b in range(B):
            cam_rot = R[b]
            cam_pos = T[b]

            world2cam = torch.eye(4, device=self.device)
            world2cam[:3, :3] = cam_rot.T
            world2cam[:3, 3] = -cam_rot.T @ cam_pos

            if K is not None:
                proj = self.build_projection_from_K(K[b])
            else:
                proj = self.get_projection_matrix(self.fov, near=0.01, far=10.0).to(self.device)

            VP = proj @ world2cam  # (4, 4)

            verts_b = verts[b]  # (V, 3)
            print(verts_b.shape)
            ones = torch.ones((verts_b.shape[0], 1), device=verts.device, dtype=verts.dtype)
            verts_homo = torch.cat([verts_b, ones], dim=-1)  # (V, 4)

            verts_cam = verts_homo @ VP.T  # (V, 4)
            verts_ndc = verts_cam[:, :3] / verts_cam[:, 3:].clamp(min=1e-8)  # (V, 3)

            face_b = faces[b].squeeze(0) if faces[b].dim() == 3 else faces[b]  # (F, 3)
            print(face_b.shape)

            face_features = torch.ones((face_b.shape[0], 3), device=verts.device)

            rast_out = rasterize(
                verts_ndc.unsqueeze(0),        # (1, V, 3)
                face_b.unsqueeze(0),           # (1, F, 3)
                face_features.unsqueeze(0),    # (1, F, 3)
                self.image_size,
                self.image_size
            )

            mask = rast_out['mask'][0].float().unsqueeze(-1).repeat(1, 1, 3)
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

    def build_projection_from_K(self, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = self.image_size, self.image_size

        proj = torch.tensor([
            [2 * fx / w, 0,          (w - 2 * cx) / w, 0],
            [0,          2 * fy / h, (h - 2 * cy) / h, 0],
            [0,          0,         -1,               -0.2],
            [0,          0,         -1,                0]
        ], dtype=torch.float32, device=self.device)
        return proj
