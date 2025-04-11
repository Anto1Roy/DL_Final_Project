import math
import torch
import torch.nn as nn
import kaolin.ops.mesh
from kaolin.render.mesh import rasterize

class KaolinRenderer(nn.Module):
    def __init__(self, width=640, height=480, fov=60.0, device='cuda'):
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov
        self.device = device

    def forward(self, verts, faces, R_obj, T_obj, K, cam_extrinsics):
        """
        Renders the mesh specified by `verts` and `faces` from the viewpoint given by the camera parameters.
        
        Args:
            verts: Tensor of shape (V, 3) representing the object mesh vertices.
            faces: Tensor of shape (F, 3) containing face indices into verts.
            R_obj: Predicted object pose rotation matrix (global, shape (3,3)).
            T_obj: Predicted object pose translation vector (global, shape (3,)).
            K: Camera intrinsics (tensor of shape (3,3)).
            cam_extrinsics (optional): Dictionary with keys:
                  "R_w2c": camera rotation from world to camera (tensor (3,3))
                  "t_w2c": camera translation (tensor (3,))
                  If provided, it is used to compute the projection.
        
        Returns:
            A rendered image tensor of shape (1, H, W, 3).
        """
        # Ensure inputs are batched if necessary.
        if verts.dim() == 2:
            verts = verts.unsqueeze(0)  # (1, V, 3)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)  # (1, F, 3)
        if R_obj.dim() == 2:
            R_obj = R_obj.unsqueeze(0)  # (1, 3, 3)
        if T_obj.dim() == 1:
            T_obj = T_obj.unsqueeze(0)  # (1, 3)

        verts_world = torch.bmm(verts, R_obj.transpose(1, 2)) + T_obj.unsqueeze(1)  # (B, V, 3)

        R_w2c = cam_extrinsics["R_w2c"].to(self.device)  # (3,3)
        t_w2c = cam_extrinsics["t_w2c"].to(self.device)  # (3,)
        world2cam = torch.eye(4, device=self.device, dtype=verts.dtype)
        world2cam[:3, :3] = R_w2c
        world2cam[:3, 3] = t_w2c

        proj = self.build_projection_from_K(K)

        # Combined view-projection matrix.
        VP = proj @ world2cam  # (4,4)

        # Transform object vertices from world to camera space.
        ones = torch.ones((verts_world.shape[1], 1), device=self.device, dtype=verts.dtype)
        verts_world_homo = torch.cat([verts_world[0], ones], dim=-1)  # (V,4) for batch index 0.
        verts_cam = verts_world_homo @ VP.T  # (V,4)
        verts_cam_z = verts_cam[:, 2:3].clamp(min=1e-8)
        verts_ndc = verts_cam[:, :3] / verts_cam_z  # (V,3)
        verts_ndc = torch.nan_to_num(verts_ndc, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Now gather projected vertices for faces.
        face_b = faces[0].to(self.device)  # (F, 3)
        # Get 2D positions (x, y) from NDC.
        face_vertices_image = verts_ndc[face_b][:, :, :2].unsqueeze(0).float()  # (1, F, 3, 2)
        face_vertices_z = verts_ndc[face_b][:, :, 2].unsqueeze(0).float()       # (1, F, 3)
        
        # --- Lambertian shading ---
        verts_cam_space = verts_world[0] @ world2cam[:3, :3].T + world2cam[:3, 3]  # (V,3)
        tri_verts = verts_cam_space[face_b]  # (F, 3, 3)
        v1 = tri_verts[:, 1] - tri_verts[:, 0]
        v2 = tri_verts[:, 2] - tri_verts[:, 0]
        normals = torch.cross(v1, v2, dim=-1)  # (F,3)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        # Light is assumed to come from the camera (z-axis)
        light_dir = torch.tensor([0, 0, 1.0], device=self.device, dtype=verts.dtype)
        lambert = torch.clamp((normals @ light_dir), min=0.0)  # (F,)
        lambert = lambert.view(1, -1, 1, 1)
        
        # Base color (e.g., red)
        base_color = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=verts.dtype).view(1, 1, 3, 1)
        face_features = base_color.repeat(1, face_b.shape[0], 1, 1) * lambert

        # Rasterize faces.
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
        # K is assumed to be (3,3).
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        w, h = self.width, self.height

        proj = torch.tensor([
            [2 * fx / w, 0,          (w - 2 * cx) / w, 0],
            [0,          2 * fy / h, (h - 2 * cy) / h, 0],
            [0,          0,         -1,               -0.2],
            [0,          0,         -1,                0]
        ], dtype=torch.float32, device=self.device)
        return proj
