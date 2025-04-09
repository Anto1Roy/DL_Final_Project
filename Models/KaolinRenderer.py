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

    def forward(self, verts, faces, R, T, K=None):
        
        # Ensure inputs are batched
        if verts.dim() == 2:
            verts = verts.unsqueeze(0)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() == 1:
            T = T.unsqueeze(0)

        # Camera transformation
        cam_rot = R[0]  # (3,3)
        cam_pos = T[0]  # (3,)
        world2cam = torch.eye(4, device=self.device, dtype=verts.dtype)
        world2cam[:3, :3] = cam_rot.T
        world2cam[:3, 3] = -cam_rot.T @ cam_pos

        if K is not None:
            proj = self.build_projection_from_K(K.squeeze(0))
        else:
            proj = self.get_projection_matrix(self.fov, near=0.01, far=10.0).to(self.device)

        VP = proj @ world2cam  # (4,4)
        verts_b = verts[0].to(self.device)  # (V, 3)
        face_b = faces[0].to(self.device)   # (F, 3)

        # Convert to homogeneous coordinates
        ones = torch.ones((verts_b.shape[0], 1), device=self.device, dtype=verts.dtype)
        verts_homo = torch.cat([verts_b, ones], dim=-1)  # (V,4)
        verts_cam = verts_homo @ VP.T  # (V,4)
        verts_ndc = verts_cam[:, :3] / verts_cam[:, 3:].clamp(min=1e-8)
        verts_ndc = torch.nan_to_num(verts_ndc, nan=0.0, posinf=0.0, neginf=0.0)

        # Get 2D positions and depth for the faces
        face_vertices_image = verts_ndc[face_b][:, :, :2].unsqueeze(0).float()  # (1, F, 3, 2)
        face_vertices_z = verts_ndc[face_b][:, :, 2].unsqueeze(0).float()       # (1, F, 3)

        # ----- Improved Shading with Lambertian Term -----
        # Compute face normals from triangle vertices in camera space.
        tri_verts = verts_b[face_b]  # (F, 3, 3)
        v1 = tri_verts[:, 1] - tri_verts[:, 0]
        v2 = tri_verts[:, 2] - tri_verts[:, 0]
        normals = torch.cross(v1, v2, dim=-1)  # (F, 3)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Define a light direction (e.g., coming from the camera)
        light_dir = torch.tensor([0, 0, 1.0], device=self.device, dtype=verts.dtype)
        lambert = torch.clamp((normals @ light_dir), min=0.0)  # (F,)
        # Reshape and broadcast for each face and each vertex
        lambert = lambert.view(1, -1, 1, 1)  # (1, F, 1, 1)
        
        # Define a base color. For example, red:
        base_color = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=verts.dtype).view(1, 1, 3, 1)
        # Repeat for all faces:
        face_features = base_color.repeat(1, face_b.shape[0], 1, 1) * lambert
        # ---------------------------------------------------

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
        # Convert intrinsic parameters to scalars with .item()
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

    def render_mesh(self, verts, faces, R, T, K, background="white", resolution=None):
        if resolution:
            self.height, self.width = resolution

        image_tensor = self.forward(verts, faces, R, T, K)
        return image_tensor[0].permute(2, 0, 1)  # (3, H, W)

    def render_scene(self, meshes, K, background="white", resolution=None):
        if resolution:
            self.height, self.width = resolution

        if background == "white":
            bg = torch.ones((1, self.height, self.width, 3), device=self.device, dtype=torch.float32)
        else:
            bg = background.unsqueeze(0).to(self.device)

        composite = bg.clone()
        alpha = 0.5

        for mesh in meshes:
            verts = mesh['verts']
            faces = mesh['faces']
            R = mesh['R']
            T = mesh['T']
            rendered = self.forward(verts, faces, R, T, K)
            composite = alpha * rendered + (1 - alpha) * composite

        return composite[0].permute(2, 0, 1)
