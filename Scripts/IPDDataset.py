import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IPDValidationDataset(Dataset):
    def __init__(self, root_dir, cam_ids, transform=None):
        self.root_dir = root_dir
        self.cam_ids = cam_ids  # ["cam1", "cam2", "cam3"]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.samples = self._load_index()

    def _load_index(self):
        samples = []
        for scene_id in sorted(os.listdir(os.path.join(self.root_dir, "val"))):
            scene_path = os.path.join(self.root_dir, "val", scene_id)
            for cam_id in self.cam_ids:
                cam_gt_path = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                if not os.path.exists(cam_gt_path):
                    continue
                with open(cam_gt_path) as f:
                    gt_data = json.load(f)
                for frame_id, objects in gt_data.items():
                    for obj in objects:
                        samples.append((scene_id, frame_id, obj["obj_id"]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, frame_id, obj_id = self.samples[idx]
        img_list, depth_list, aolp_list, dolp_list = [], [], [], []
        Ks = []

        for cam_id in self.cam_ids:
            base = os.path.join(self.root_dir, "val", scene_id)
            fid = frame_id.zfill(6)

            rgb_path = os.path.join(base, f"rgb_{cam_id}", f"{fid}.png")
            depth_path = os.path.join(base, f"depth_{cam_id}", f"{fid}.png")
            aolp_path = os.path.join(base, f"aolp_{cam_id}", f"{fid}.png")
            dolp_path = os.path.join(base, f"dolp_{cam_id}", f"{fid}.png")

            rgb = self.transform(cv2.imread(rgb_path))
            depth = self.transform(cv2.imread(depth_path, -1).astype(np.float32) / 65535.0)
            aolp = self.transform(cv2.imread(aolp_path, 0) / 255.0)
            dolp = self.transform(cv2.imread(dolp_path, 0) / 255.0)

            with open(os.path.join(base, f"scene_camera_{cam_id}.json")) as f:
                K = np.array(json.load(f)[fid]["cam_K"]).reshape(3, 3)
                Ks.append(torch.tensor(K, dtype=torch.float32))

            img_list.append(rgb)
            depth_list.append(depth)
            aolp_list.append(aolp)
            dolp_list.append(dolp)

        with open(os.path.join(base, f"scene_gt_{cam_id}.json")) as f:
            gt = json.load(f)[fid][0]  # we use first GT match for now
            R = torch.tensor(gt["cam_R_m2c"]).reshape(3, 3)
            t = torch.tensor(gt["cam_t_m2c"]) / 1000.0  # convert mm to meters

        return (
            torch.stack(img_list),
            torch.stack(depth_list),
            torch.stack(aolp_list),
            torch.stack(dolp_list),
            torch.stack(Ks),
            R,
            t
        )
