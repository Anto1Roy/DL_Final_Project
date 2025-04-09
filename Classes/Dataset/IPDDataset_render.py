import os
import sys
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import trimesh

# --- Local Import ---
sys.path.append(os.getcwd())
from Classes.Dataset.StreamingFileManager import StreamingFileManager

class IPDDatasetMounted(Dataset):
    def __init__(self, remote_base_url, cam_ids, modalities=["rgb", "depth"], split="val", transform=None, models_dir=None, allowed_obj_ids=None, allowed_scene_ids=None):
        self.img_size = (640, 480)
        self.file_manager = StreamingFileManager(remote_base_url)
        self.cam_ids = cam_ids
        self.modalities = modalities
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        self.remote_base = os.path.expanduser(remote_base_url)
        self.allowed_obj_ids = set(allowed_obj_ids) if allowed_obj_ids is not None else None
        self.allowed_scene_ids = set(allowed_scene_ids) if allowed_scene_ids is not None else None
        self.samples = self._load_index()
        self.models_dir = models_dir or os.path.join(remote_base_url, "models")

    def _load_index(self):
        samples = []
        scene_dir = os.path.join(self.remote_base, self.split)
        for scene_id in sorted(os.listdir(scene_dir)):
            if self.allowed_scene_ids is not None and scene_id not in self.allowed_scene_ids:
                continue

            for cam_id in self.cam_ids:
                local_path = os.path.join(scene_dir, scene_id, f"scene_gt_{cam_id}.json")
                if not os.path.exists(local_path):
                    continue
                with open(local_path) as f:
                    gt_data = json.load(f)
                for frame_id, objects in gt_data.items():
                    for obj in objects:
                        if self.allowed_obj_ids is None or obj["obj_id"] in self.allowed_obj_ids:
                            samples.append((scene_id, frame_id, obj["obj_id"], cam_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, frame_id, _, _ = self.samples[idx]
        fid = frame_id.zfill(6)
        fid_key = str(int(frame_id))
        base_remote = f"{self.split}/{scene_id}"

        x_dict_views = []
        Ks = []
        valid_views = []

        for cam_id in self.cam_ids:
            view_dict = {}
            view_valid = True

            for modality in self.modalities:
                ext = 'png' if modality != 'rgb' else ('png' if self.split == "val" else 'jpg')
                remote_path = f"{base_remote}/{modality}_{cam_id}/{fid}.{ext}"
                try:
                    view_dict[modality] = self.read_img(remote_path, modality)
                except Exception:
                    view_valid = False
                    break

            if view_valid:
                try:
                    cam_idx = int(cam_id.replace("cam", ""))
                    cam_cfg = json.load(open(self.file_manager.get(f"ipd/camera_cam{cam_idx}.json")))
                    fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
                    W_orig, H_orig = cam_cfg["width"], cam_cfg["height"]
                    W_new, H_new = self.img_size
                    scale_x, scale_y = W_new / W_orig, H_new / H_orig

                    K = torch.tensor([
                        [fx * scale_x, 0, cx * scale_x],
                        [0, fy * scale_y, cy * scale_y],
                        [0, 0, 1]
                    ], dtype=torch.float32)

                    Ks.append(K)
                    x_dict_views.append(view_dict)
                    valid_views.append(cam_id)
                except Exception:
                    continue

        if len(x_dict_views) != len(self.cam_ids):
            return None

        Ks = torch.stack(Ks)

        # === Ground truth ===
        primary_cam = valid_views[0]
        gt_all = json.load(open(self.file_manager.get(f"{base_remote}/scene_gt_{primary_cam}.json")))[fid_key]

        gt_poses = []
        # available_cads = []
        model_points_by_id = {}

        for obj in gt_all:
            obj_id = int(obj["obj_id"])
            R = torch.tensor(obj["cam_R_m2c"], dtype=torch.float32).reshape(3, 3)
            t = torch.tensor(obj["cam_t_m2c"], dtype=torch.float32).reshape(3) / 1000.0
            instance_id = obj.get("instance_id", obj_id)  # optional, fallback to obj_id

            model_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")

            if obj_id not in model_points_by_id:
                model_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")
                model_points_by_id[obj_id] = self.load_cad_model_points(model_path, num_points=500)

            obj_id = torch.tensor(obj["obj_id"], dtype=torch.float32)

            gt_poses.append({
                "R": R,
                "t": t,
                "obj_id": obj_id,
                "instance_id": instance_id,
            })

            # available_cads.append({
            #     "obj_id": obj_id,
            #     "verts": torch.tensor(mesh.vertices, dtype=torch.float32),
            #     "faces": torch.tensor(mesh.faces, dtype=torch.long),
            # })

        return {
            "X": {
                "views": x_dict_views,
                "K": Ks,
                "model_points_by_id": model_points_by_id
                # "available_cads": available_cads,
            },
            "Y": {
                "gt_poses": gt_poses
            }
        }


    def read_img(self, remote_path, modality):
        local_path = self.file_manager.get(remote_path)

        if modality == "rgb":
            img = cv2.imread(local_path)
            if img is None:
                raise FileNotFoundError(f"RGB image not found at {remote_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tensor = self.transform(img[:, :, None])
        else:
            flag = cv2.IMREAD_UNCHANGED if modality == "depth" else 0
            img = cv2.imread(local_path, flag)
            if img is None:
                raise FileNotFoundError(f"{modality.upper()} image not found at {remote_path}")
            norm = 65535.0 if modality == "depth" else 255.0
            tensor = self.transform((img.astype(np.float32) / norm)[:, :, None])
        return tensor
    
    def load_cad_model_points(self, mesh_path, num_points=500):
        mesh = trimesh.load(mesh_path, force='mesh')

        # Uniformly sample points on the surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        return torch.tensor(points, dtype=torch.float32)  # (N, 3)
