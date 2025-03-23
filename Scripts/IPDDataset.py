import os
import sys
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --- Local Import ---
sys.path.append(os.getcwd())

class IPDValidationDataset(Dataset):
    def __init__(self, root_dir, cam_ids, modalities=["rgb", "depth"], split="val", transform=None):
        self.root_dir = root_dir
        self.cam_ids = cam_ids
        self.modalities = modalities
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),  # <-- Resize here
            transforms.ToTensor()
        ])

        self.samples = self._load_index()

    def _load_index(self):
        samples = []
        val_path = os.path.join(self.root_dir, self.split)
        print(f"Loading {self.split} data from {val_path}")
        for scene_id in sorted(os.listdir(val_path)):
            scene_path = os.path.join(val_path, scene_id)
            for cam_id in self.cam_ids:
                cam_gt_path = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                if not os.path.exists(cam_gt_path):
                    continue
                with open(cam_gt_path) as f:
                    gt_data = json.load(f)
                for frame_id, objects in gt_data.items():
                    for obj in objects:
                        samples.append((scene_id, frame_id, obj["obj_id"], cam_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, frame_id, obj_id, cam_id = self.samples[idx]
        base = os.path.join(self.root_dir, self.split, scene_id)
        fid = frame_id.zfill(6)
        fid_key = str(int(frame_id))


        input_modalities = []
        for modality in self.modalities:
            if modality == "rgb":
                img = cv2.imread(os.path.join(base, f"rgb_{cam_id}", f"{fid}.png"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert RGB to grayscale
                img = self.transform(img[:, :, None])  # Shape: (H, W, 1)
                input_modalities.append(img)
            elif modality == "depth":
                depth = cv2.imread(os.path.join(base, f"depth_{cam_id}", f"{fid}.png"), -1)
                input_modalities.append(self.transform((depth.astype(np.float32) / 65535.0)[:, :, None]))
            elif modality == "aolp":
                aolp = cv2.imread(os.path.join(base, f"aolp_{cam_id}", f"{fid}.png"), 0)
                input_modalities.append(self.transform((aolp / 255.0)[:, :, None]))
            elif modality == "dolp":
                dolp = cv2.imread(os.path.join(base, f"dolp_{cam_id}", f"{fid}.png"), 0)
                input_modalities.append(self.transform((dolp / 255.0)[:, :, None]))

        x = torch.cat(input_modalities, dim=0)  # shape: (C, H, W)

        # Load intrinsics from camera config (shared per cam)
        cam_idx = int(cam_id.replace("cam", ""))  # e.g., "cam2" → 2
        with open(os.path.join(self.root_dir, f"ipd/camera_cam{cam_idx}.json")) as f:
            cam_cfg = json.load(f)

        fx, fy = cam_cfg["fx"], cam_cfg["fy"]
        cx, cy = cam_cfg["cx"], cam_cfg["cy"]
        W_orig, H_orig = cam_cfg["width"], cam_cfg["height"]
        W_new, H_new = 512, 512  # Your resized image shape

        # Scale intrinsics to match new size
        scale_x = W_new / W_orig
        scale_y = H_new / H_orig

        K = np.array([
            [fx * scale_x, 0,     cx * scale_x],
            [0,     fy * scale_y, cy * scale_y],
            [0,     0,     1]
        ], dtype=np.float32)

        K = torch.tensor(K, dtype=torch.float32)

        with open(os.path.join(base, f"scene_gt_{cam_id}.json")) as f:
            gt = json.load(f)[fid_key][0]  # take first GT match
            R = torch.tensor(gt["cam_R_m2c"]).reshape(3, 3)
            t = torch.tensor(gt["cam_t_m2c"]) / 1000.0  # mm to meters

        return x, R, t, K
