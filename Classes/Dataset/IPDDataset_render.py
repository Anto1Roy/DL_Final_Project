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

        x_dict = {modality: [] for modality in self.modalities}
        Ks = []
        valid_views = []

        for cam_id in self.cam_ids:
            view_modalities = []
            view_valid = True
            for modality in self.modalities:
                if modality == "rgb":
                    remote_path = f"{base_remote}/{modality}_{cam_id}/{fid}.jpg"
                else:
                    remote_path = f"{base_remote}/{modality}_{cam_id}/{fid}.png"
                try:
                    img = self.read_img(remote_path, modality)
                    x_dict[modality].append(img)
                except Exception as e:
                    print(f"[WARN] Skipping cam {cam_id} for sample {idx} due to missing {modality} ({e})")
                    view_valid = False
                    break

            if not view_valid:
                # remove incomplete view
                for modality in self.modalities:
                    if len(x_dict[modality]) > 0:
                        x_dict[modality].pop()
                continue

            try:
                cam_idx = int(cam_id.replace("cam", ""))
                local_cam_json = self.file_manager.get(f"ipd/camera_cam{cam_idx}.json")
                with open(local_cam_json) as f:
                    cam_cfg = json.load(f)

                fx, fy = cam_cfg["fx"], cam_cfg["fy"]
                cx, cy = cam_cfg["cx"], cam_cfg["cy"]
                W_orig, H_orig = cam_cfg["width"], cam_cfg["height"]
                W_new, H_new = self.img_size
                scale_x = W_new / W_orig
                scale_y = H_new / H_orig

                K = np.array([
                    [fx * scale_x, 0,     cx * scale_x],
                    [0,     fy * scale_y, cy * scale_y],
                    [0,     0,     1]
                ], dtype=np.float32)
                Ks.append(torch.tensor(K, dtype=torch.float32))

                valid_views.append(cam_id)
            except Exception as e:
                print(f"[WARN] Camera intrinsics missing for cam {cam_id}: {e}")

        # use first valid cam_id to get GT (assumes identical GT across views)
        if not valid_views:
            return None

        primary_cam = valid_views[0]
        local_gt_json = self.file_manager.get(f"{base_remote}/scene_gt_{primary_cam}.json")
        with open(local_gt_json) as f:
            gt_all = json.load(f)[fid_key]

        R_gt, t_gt, instance_ids, verts_all, faces_all = [], [], [], [], []
        for obj in gt_all:
            obj_id = obj["obj_id"]
            R = torch.tensor(obj["cam_R_m2c"], dtype=torch.float32).reshape(3, 3)
            t = torch.tensor(obj["cam_t_m2c"], dtype=torch.float32).reshape(3) / 1000.0
            R_gt.append(R)
            t_gt.append(t)

            model_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")
            mesh = trimesh.load(model_path, process=False)
            verts_all.append(torch.tensor(mesh.vertices, dtype=torch.float32))
            faces_all.append(torch.tensor(mesh.faces, dtype=torch.long))
            instance_ids.append(obj_id)

        Ks = torch.stack(Ks)  # (V, 3, 3)

        return x_dict, Ks, R_gt, t_gt, instance_ids, list(zip(verts_all, faces_all))

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
