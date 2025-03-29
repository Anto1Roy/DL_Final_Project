import os
import sys
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# --- Local Import ---s
sys.path.append(os.getcwd())

from StreamingFileManager import StreamingFileManager

class IPDValidationDatasetMounted(Dataset):
    def __init__(self, remote_base_url, cam_ids, modalities=["rgb", "depth"], split="val", transform=None):
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
        self.remote_base = remote_base_url
        self.samples = self._load_index()

    def _load_index(self):
        samples = []
        val_list_url = f"{self.remote_base}/{self.split}/"

        for scene_id in sorted(os.listdir(os.path.join(self.remote_base, self.split))):  # local mirror just to index
            for cam_id in self.cam_ids:
                local_path = os.path.join(self.remote_base, self.split, scene_id, f"scene_gt_{cam_id}.json")
                if not os.path.exists(local_path):
                    continue
                with open(local_path) as f:
                    gt_data = json.load(f)
                for frame_id, objects in gt_data.items():
                    for obj in objects:
                        samples.append((scene_id, frame_id, obj["obj_id"], cam_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, frame_id, obj_id, cam_id = self.samples[idx]
        fid = frame_id.zfill(6)
        fid_key = str(int(frame_id))

        base_remote = f"{self.split}/{scene_id}"

        input_modalities = []
        for modality in self.modalities:
            remote_path = f"{base_remote}/{modality}_{cam_id}/{fid}.png"
            img = self.read_img(remote_path, modality)
            input_modalities.append(img)

        x_dict = {modality: tensor for modality, tensor in zip(self.modalities, input_modalities)}

        # Intrinsics
        cam_idx = int(cam_id.replace("cam", ""))
        local_cam_json = self.file_manager.get(f"ipd/camera_cam{cam_idx}.json")
        with open(local_cam_json) as f:
            cam_cfg = json.load(f)
        self.file_manager.remove(local_cam_json)

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
        K = torch.tensor(K, dtype=torch.float32)

        # Pose
        local_gt_json = self.file_manager.get(f"{base_remote}/scene_gt_{cam_id}.json")
        with open(local_gt_json) as f:
            gt = json.load(f)[fid_key][0]
        self.file_manager.remove(local_gt_json)

        R = torch.tensor(gt["cam_R_m2c"]).reshape(3, 3)
        t = torch.tensor(gt["cam_t_m2c"]) / 1000.0

        return x_dict, R, t, K

    def read_img(self, remote_path, modality):
        local_path = self.file_manager.get(remote_path)

        if modality == "rgb":
            img = cv2.imread(local_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tensor = self.transform(img[:, :, None])
        else:
            flag = cv2.IMREAD_UNCHANGED if modality == "depth" else 0
            img = cv2.imread(local_path, flag)
            norm = 65535.0 if modality == "depth" else 255.0
            tensor = self.transform((img.astype(np.float32) / norm)[:, :, None])

        self.file_manager.remove(local_path)
        return tensor