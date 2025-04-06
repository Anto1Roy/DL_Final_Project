# train_candidate_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import yaml
import sys

# Local
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Classes.EarlyStopping import EarlyStopping
from Models.CandidatePose import CandidatePoseModel
from Models.helpers import R_to_quat

def compute_detection_loss(output, gt_quat, gt_trans, gt_class_id):
    
    loss_q = F.mse_loss(output["quat"].mean(dim=(1, 2, 3)), gt_quat)
    loss_t = F.mse_loss(output["trans"].mean(dim=(1, 2, 3)), gt_trans)
    class_logits = output["class_scores"].mean(dim=(1, 2, 3))  # B x C
    loss_c = F.cross_entropy(class_logits, gt_class_id)
    return loss_q + loss_t + loss_c, loss_q.item(), loss_t.item(), loss_c.item()

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"]["modality"]
    split = config["dataset"]["train_split"]
    batch_size = config["training"]["batch_size"]
    device = config["training"]["device"]

    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities, split=split)
    val_split = 0.2
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split_idx = int(val_split * num_samples)
    train_sampler = SubsetRandomSampler(indices[split_idx:])
    val_sampler = SubsetRandomSampler(indices[:split_idx])

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    sensory_channels = {mod: 1 for mod in modalities}
    model = CandidatePoseModel(sensory_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x_dict, R_gt_list, t_gt_list, _, _, _, instance_ids = batch
            gt_class_id = torch.tensor([ids[0] for ids in instance_ids], device=device)

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            quat = torch.stack([R_to_quat(R) for R in R_gt_list], dim=0).to(device)
            trans = torch.stack(t_gt_list, dim=0).to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(x_dict)
                loss, lq, lt, lc = compute_detection_loss(output, quat, trans, gt_class_id)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    torch.save(model.state_dict(), "./weights/candidate_pose_model.pth")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
