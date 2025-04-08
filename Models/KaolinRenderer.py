# --- Imports ---
import math
import torch
import torch.nn as nn

from kaolin.render.mesh import rasterize


class KaolinRenderer(nn.Module):
    def __init__(self, width=640, height=480, fov=60.0, device='cuda'):
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov
        self.device = device

    def forward(self, verts, faces, R, T, K=None):
        # Assumes input shapes:
        # verts: (1, V, 3), faces: (1, F, 3)
        # R: (1, 3, 3), T: (1, 3)
        # Returns: mask: (1, H, W, 3)

        cam_rot = R[0]  # (3, 3)
        cam_pos = T[0]  # (3,)

        world2cam = torch.eye(4, device=self.device, dtype=verts.dtype)
        world2cam[:3, :3] = cam_rot.T
        world2cam[:3, 3] = -cam_rot.T @ cam_pos

        if K is not None:
            proj = self.build_projection_from_K(K.squeeze(0))
        else:
            proj = self.get_projection_matrix(self.fov, near=0.01, far=10.0).to(self.device)

        VP = proj @ world2cam  # (4, 4)

        verts_b = verts[0].to(self.device)  # (V, 3)
        face_b = faces[0].to(self.device)   # (F, 3)

        ones = torch.ones((verts_b.shape[0], 1), device=self.device, dtype=verts.dtype)
        verts_homo = torch.cat([verts_b, ones], dim=-1)  # (V, 4)
        verts_cam = verts_homo @ VP.T  # (V, 4)
        verts_ndc = verts_cam[:, :3] / verts_cam[:, 3:].clamp(min=1e-8)  # (V, 3)
        verts_ndc = torch.nan_to_num(verts_ndc, nan=0.0, posinf=0.0, neginf=0.0)

        face_vertices_image = verts_ndc[face_b][:, :, :2].unsqueeze(0).float()  # (1, F, 3, 2)
        face_vertices_z = verts_ndc[face_b][:, :, 2].unsqueeze(0).float()       # (1, F, 3)

        # gray scaled RGB
        face_features = torch.ones((1, face_b.shape[0], 3, 1), device=self.device, dtype=torch.float32)

        rast_out, _ = rasterize(
            height=self.height,
            width=self.width,
            face_vertices_image=face_vertices_image,
            face_vertices_z=face_vertices_z,
            face_features=face_features
        )

        return rast_out  # (1, H, W, 3)
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
        # K = K.view(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = self.width, self.height

        proj = torch.tensor([
            [2 * fx / w, 0,          (w - 2 * cx) / w, 0],
            [0,          2 * fy / h, (h - 2 * cy) / h, 0],
            [0,          0,         -1,               -0.2],
            [0,          0,         -1,                0]
        ], dtype=torch.float32, device=self.device)
        return proj
