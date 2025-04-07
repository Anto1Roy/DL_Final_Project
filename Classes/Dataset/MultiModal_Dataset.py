# data/ipd_multimodal_dataset.py
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, data_root, cam_ids, modalities=["rgb", "depth", "aolp", "dolp"], transform=None):
        self.img_size = (640, 480)  # or any desired size
        self.cam_ids = cam_ids
        self.modalities = modalities
        self.data_root = data_root
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        self.samples = self._load_index()

    def _load_index(self):
        samples = []
        for scene_id in sorted(os.listdir(self.data_root)):
            for cam_id in self.cam_ids:
                local_path = os.path.join(self.data_root, scene_id, f"scene_gt_{cam_id}.json")
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

        base_path = os.path.join(self.data_root, scene_id)

        input_modalities = []
        for modality in self.modalities:
            if modality == "rgb":
                img_path = f"{base_path}/{modality}_{cam_id}/{fid}.jpg"
            else:
                img_path = f"{base_path}/{modality}_{cam_id}/{fid}.png"
            img = self.read_img(img_path, modality)
            input_modalities.append(img)

        # Apply transform to the images (e.g., resizing and normalization)
        x_dict = {modality: tensor for modality, tensor in zip(self.modalities, input_modalities)}

        # Load camera intrinsic parameters
        cam_params = self.load_camera_intrinsics(cam_id)
        K = torch.tensor(cam_params, dtype=torch.float32)

        # Load object pose
        pose = self.load_object_pose(base_path, fid)
        R = torch.tensor(pose["cam_R_m2c"]).reshape(3, 3)
        t = torch.tensor(pose["cam_t_m2c"]) / 1000.0  # Convert to meters

        return x_dict, R, t, K

    def read_img(self, path, modality):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED if modality == "depth" else 0)
        if modality == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = self.transform(img)
        return tensor

    def load_camera_intrinsics(self, cam_id):
        cam_cfg_path = os.path.join(self.data_root, f"camera_{cam_id}.json")
        with open(cam_cfg_path) as f:
            cam_cfg = json.load(f)
        fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
        return [fx, fy, cx, cy]  # Simplified for now

    def load_object_pose(self, base_path, fid):
        pose_path = os.path.join(base_path, f"scene_gt.json")
        with open(pose_path) as f:
            gt = json.load(f)
        return gt[str(fid)][0]
