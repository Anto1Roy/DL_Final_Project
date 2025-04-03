import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from itertools import islice

class StreamingIPDIterableDataset(IterableDataset):
    def __init__(self, stream_data, modalities=["rgb", "depth"], transform=None, skip=0):
        self.stream_data = islice(stream_data, skip, None)
        self.modalities = modalities
        self.img_size = (640, 480)
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __iter__(self):
        for ex in self.stream_data:
            x_dict = {}

            for modality in self.modalities:
                img = ex[modality]
                if isinstance(img, Image.Image):
                    img = img.convert("L") if modality == "rgb" else img
                    tensor = self.transform(img)
                    if tensor.ndim == 2:
                        tensor = tensor.unsqueeze(0)
                    x_dict[modality] = tensor
                else:
                    continue

            # Intrinsics
            fx, fy = 1066.778, 1067.487
            cx, cy = 312.9869, 241.3109
            W_orig, H_orig = 640, 480
            scale_x = self.img_size[0] / W_orig
            scale_y = self.img_size[1] / H_orig

            K = np.array([
                [fx * scale_x, 0,     cx * scale_x],
                [0,     fy * scale_y, cy * scale_y],
                [0,     0,     1]
            ], dtype=np.float32)
            K = torch.tensor(K, dtype=torch.float32)

            gt = ex["annotations"][0]
            R = torch.tensor(gt["cam_R_m2c"]).reshape(3, 3).float()
            t = torch.tensor(gt["cam_t_m2c"]).float() / 1000.0

            yield x_dict, R, t, K
