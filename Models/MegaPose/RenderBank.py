import math
import random
import torch
import torch.nn as nn
from Models.KaolinRenderer import KaolinRenderer

class RenderBank:
    """
    A class to manage CAD render templates for a set of objects.
    The render bank uses KaolinRenderer to pre-render images (and corresponding 
    canonical poses) for each CAD model.
    """
    def __init__(self, cad_models, num_templates=20, renderer_params=None, distance=3.0, device='cuda', K=None):
        """
        Args:
            cad_models: dict mapping object id to CAD model data.
                        Each value should be a dict with keys "verts" and "faces".
            num_templates (int): Number of render templates per CAD model.
            renderer_params (dict): Optional parameters for the renderer (width, height, fov).
            distance (float): Fixed camera-to-object distance to use when rendering.
            device (str): Device to use for rendering.
            K: Optional intrinsic parameter(s). It can be a single tensor of shape (3,3)
               or a list of such tensors. When provided, the renderer uses one intrinsic
               for each template (randomly selected if multiple are provided).
        """
        self.cad_models = cad_models
        self.num_templates = num_templates
        self.distance = distance
        self.device = device

        # If K is provided, store it as a list for easy random selection.
        if K is not None:
            if isinstance(K, list):
                self.K_list = [k.to(self.device) for k in K]
            else:
                self.K_list = [K.to(self.device)]
        else:
            self.K_list = None

        # Renderer parameters with defaults.
        if renderer_params is None:
            renderer_params = {"width": 640, "height": 480, "fov": 60.0}
        
        self.renderer = KaolinRenderer(width=renderer_params.get("width", 640),
                                       height=renderer_params.get("height", 480),
                                       fov=renderer_params.get("fov", 60.0),
                                       device=device)
        # Build templates using the renderer.
        self.templates = self._build_templates()
    
    def _build_templates(self):
        templates = {}
        for obj_id, cad_model in self.cad_models.items():
            # Expecting cad_model to be a dict with "verts" and "faces".
            verts = cad_model['verts']  # Tensor of shape (V, 3)
            faces = cad_model['faces']  # Tensor of shape (F, 3)
            # Move vertices and faces to the renderer's device and add batch dimension.
            verts = verts.to(self.device)
            faces = faces.to(self.device)
            verts = verts.unsqueeze(0)  # Now shape (1, V, 3)
            faces = faces.unsqueeze(0)  # Now shape (1, F, 3)
            
            template_list = []
            for i in range(self.num_templates):
                R, T = self._get_random_camera_pose(self.distance, self.device)
                # If intrinsic(s) are provided, randomly choose one.
                if self.K_list is not None:
                    current_K = random.choice(self.K_list)
                    if current_K.dim() == 2:
                        current_K = current_K.unsqueeze(0)  # Make it (1,3,3)
                else:
                    current_K = None
                # Render the CAD model using the renderer.
                cad_render = self.renderer(verts, faces, R, T, K=current_K)
                # Build a 4x4 render pose matrix.
                render_pose = torch.eye(4, device=self.device, dtype=torch.float32)
                render_pose[:3, :3] = R[0]
                render_pose[:3, 3] = T[0]
                template_list.append((cad_render, render_pose))
            templates[obj_id] = template_list
        return templates
    
    def _get_random_camera_pose(self, d, device):
        """
        Generate a random camera pose for rendering.
        
        Args:
            d (float): Fixed distance of the camera from the object.
            device (str): Device for the resulting tensors.
        
        Returns:
            R: Rotation matrix, shape (1, 3, 3)
            T: Translation vector, shape (1, 3)
        """
        # Sample random spherical coordinates.
        theta = 2 * math.pi * torch.rand(1).item()  # Azimuth in [0, 2pi]
        phi = (math.pi / 2) * torch.rand(1).item()      # Elevation in [0, pi/2]
        # Compute camera position in Cartesian coordinates.
        x = d * math.sin(phi) * math.cos(theta)
        y = d * math.sin(phi) * math.sin(theta)
        z = d * math.cos(phi)
        T = torch.tensor([x, y, z], device=device, dtype=torch.float32)
        # Compute the camera's viewing direction (from object center at origin to camera).
        z_axis = T / (T.norm() + 1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        x_axis = torch.cross(up, z_axis)
        if x_axis.norm() < 1e-8:
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        else:
            x_axis = x_axis / (x_axis.norm() + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / (y_axis.norm() + 1e-8)
        # Form the rotation matrix such that the camera looks toward the origin.
        R = torch.stack([x_axis, y_axis, z_axis], dim=0)  # Shape (3, 3)
        R = R.unsqueeze(0)  # Shape (1, 3, 3)
        T = T.unsqueeze(0)  # Shape (1, 3)
        return R, T
    
    def get_templates(self, obj_id):
        """Return render templates for the given CAD model id."""
        return self.templates.get(obj_id, [])