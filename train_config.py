import torch
from torch.utils.data import DataLoader
from dataset import IPDValidationDataset
from model import PoseEstimationModel
import torch.nn as nn

from tqdm import tqdm
import os
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_root = config["dataset"]["root_dir"]
cam_ids = config["dataset"]["cam_ids"]
batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
num_workers = config["training"]["num_workers"]
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

lr_rot = config["optim"]["lr_rot"]
lr_trans = config["optim"]["lr_trans"]

# --- Dataset ---
dataset = IPDValidationDataset(data_root, cam_ids)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Model ---
model = PoseEstimationModel()
model.to(device)

# Separate pose head layers into rotation and translation optimizers
rot_params = []
trans_params = []

# Assuming pose_head[2] is Linear(128, 7), we separate first 4 dims (rotation), last 3 (translation)
pose_head = model.pose_head[2]
assert isinstance(pose_head, nn.Linear)
rot_params += [pose_head.weight[:4], pose_head.bias[:4]]
trans_params += [pose_head.weight[4:], pose_head.bias[4:]]

# Add shared head + fusion + transformer + encoders to both
shared_params = list(model.pose_head[0].parameters()) + \
                list(model.rgb_branch.parameters()) + \
                list(model.depth_branch.parameters()) + \
                list(model.aolp_branch.parameters()) + \
                list(model.dolp_branch.parameters()) + \
                list(model.fusion_proj.parameters()) + \
                list(model.view_transformer.parameters())

rot_optimizer = torch.optim.Adam(rot_params + shared_params, lr=lr_rot)
trans_optimizer = torch.optim.Adam(trans_params + shared_params, lr=lr_trans)

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, depths, aolps, dolps, K, R_gt, t_gt = [x.to(device) for x in batch]

        pred_R, pred_t, rot_loss, trans_loss = model(imgs, depths, aolps, dolps, K, R_gt, t_gt)

        rot_optimizer.zero_grad()
        trans_optimizer.zero_grad()

        rot_loss.backward(retain_graph=True)
        trans_loss.backward()

        rot_optimizer.step()
        trans_optimizer.step()

        total_loss += (rot_loss + trans_loss).item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), f"pose_model_epoch{epoch+1}.pth")
