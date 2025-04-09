import torch
import matplotlib.pyplot as plt
import trimesh  # pip install trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftPhongShader,
    PointLights
)
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the .ply mesh using trimesh ---
ply_path = "path_to_mesh.ply"  # Replace with your .ply file path.
if not os.path.exists(ply_path):
    raise FileNotFoundError(f"Make sure the file exists: {ply_path}")

# Load the mesh with trimesh
mesh_trimesh = trimesh.load(ply_path, process=False)
print("Mesh loaded from .ply:")
print("Vertices shape:", mesh_trimesh.vertices.shape)
print("Faces shape:", mesh_trimesh.faces.shape)

# Convert vertices and faces to torch tensors
vertices = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64, device=device)

# --- Build a Meshes object for PyTorch3D ---
# PyTorch3D expects a list of meshes (here we have one)
mesh = Meshes(verts=[vertices], faces=[faces])

# --- Setup the renderer components ---
# 1. Cameras: Here we create a simple perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# 2. Rasterization settings: Adjust image size, blur radius, etc.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 3. Lights: For example, a point light placed in front of the object.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# 4. Create a MeshRenderer object by combining a rasterizer and a shader.
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# --- Render the mesh ---
# This returns an image tensor of shape (N, H, W, 3) in [0,1]
images = renderer(mesh)

# Convert the rendered image from tensor to NumPy array and display it.
# We'll display the first (and only) image.
plt.figure(figsize=(6, 6))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.title("Rendered Mesh using PyTorch3D")
plt.axis("off")
plt.show()
