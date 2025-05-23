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
        self.img_size = (480, 640)
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
                    H_new, W_new = self.img_size  # <-- Correct unpacking of (H, W)
                    scale_x = W_new / W_orig
                    scale_y = H_new / H_orig

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
        
        scene_cam_extrinsics = []
        bbox_info_all = []

        for cam_id in valid_views:
            cam_idx = int(cam_id.replace("cam", ""))
            # === Load extrinsics ===
            scene_cam_path = self.file_manager.get(f"{base_remote}/scene_camera_cam{cam_idx}.json")
            scene_cam_data = json.load(open(scene_cam_path))[frame_id]

            R_w2c = torch.tensor(scene_cam_data["cam_R_w2c"], dtype=torch.float32).reshape(3, 3)
            t_w2c = torch.tensor(scene_cam_data["cam_t_w2c"], dtype=torch.float32).reshape(3) / 1000.0  # mm → m

            scene_cam_extrinsics.append({
                "R_w2c": R_w2c,
                "t_w2c": t_w2c
            })

            # === Load GT bounding boxes ===
            gt_info_path = self.file_manager.get(f"{base_remote}/scene_gt_info_cam{cam_idx}.json")
            bbox_data = json.load(open(gt_info_path))[frame_id]
            bbox_info_all.append(bbox_data)

        # === Intrinsics ===
        Ks = torch.stack(Ks)

        # === Ground truth for all valid views ===
        gt_poses_list = []  # This will be a list of lists, one per valid view.
        cad_lookup = {}

        for cam in valid_views:
            # Build file path based on the current view.
            gt_file_path = self.file_manager.get(f"{base_remote}/scene_gt_{cam}.json")
            # Load the GT data for this camera.
            gt_data = json.load(open(gt_file_path))
            # Use fid_key to index the current frame (make sure fid_key is a string, e.g., "1", "2", etc.)
            gt_all = gt_data.get(fid_key, [])
            
            cur_cam_gt_poses = []  # List to store GT poses for this particular view.
            for obj in gt_all:
                # Convert object information.
                obj_id = int(obj["obj_id"])
                R = torch.tensor(obj["cam_R_m2c"], dtype=torch.float32).reshape(3, 3)
                t = torch.tensor(obj["cam_t_m2c"], dtype=torch.float32).reshape(3) / 1000.0  # Conversion from mm to m
                instance_id = obj.get("instance_id", obj_id)  # Use 'instance_id' if provided; otherwise, fallback to obj_id

                # Load CAD model if not already done.
                model_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")
                if obj_id not in cad_lookup:
                    mesh = trimesh.load(model_path, force='mesh')
                    cad_lookup[obj_id] = mesh

                # Convert the object id to a tensor (if necessary).
                obj_id_tensor = torch.tensor(obj["obj_id"], dtype=torch.float32)

                cur_cam_gt_poses.append({
                    "R": R,
                    "t": t,
                    "obj_id": obj_id_tensor,
                    "instance_id": instance_id,
                })
            
            # Append this camera's GT list to the overall list.
            gt_poses_list.append(cur_cam_gt_poses)

        return {
            "X": {
                "views": x_dict_views,
                "K": Ks,
                "extrinsics": scene_cam_extrinsics,
                "cad_lookup": cad_lookup,
            },
            "Y": {
                "gt_poses": gt_poses_list,
                "bbox_info": bbox_info_all,
            }
        }



    def read_img(self, remote_path, modality):
        local_path = self.file_manager.get(remote_path)
        if modality == "rgb":
            img = cv2.imread(local_path)
            if img is None:
                raise FileNotFoundError(f"RGB image not found at {remote_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
            img = self.transform(img)  # no [:, :, None] needed!
        else:
            flag = cv2.IMREAD_UNCHANGED if modality == "depth" else 0
            img = cv2.imread(local_path, flag)
            if img is None:
                raise FileNotFoundError(f"{modality.upper()} image not found at {remote_path}")
            norm = 65535.0 if modality == "depth" else 255.0
            img = (img.astype(np.float32) / norm).clip(0, 1)
            img = self.transform(img)

        # img will now be shape (1, 480, 640)
        return img

    def load_cad_model_points(self, mesh_path, num_points=500):
        mesh = trimesh.load(mesh_path, force='mesh')

        # Uniformly sample points on the surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        return torch.tensor(points, dtype=torch.float32)  # (N, 3)
