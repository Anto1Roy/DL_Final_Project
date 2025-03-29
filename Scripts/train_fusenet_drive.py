# --- Imports ---
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import os
import yaml
import sys
from torch.cuda.amp import autocast, GradScaler

# --- Local Import ---
sys.path.append(os.getcwd())

from Models.model_fusenet_4mod import FuseNetPoseModel
from IPDDataset_drive import IPDValidationDatasetMounted

def main():

    # --- Config ---
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet_2.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]  # NEW
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    split = config["dataset"].get("split", "val")
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    num_workers = int(config["training"]["num_workers"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    lr_rot = float(config["optim"]["lr_rot"])
    lr_trans = float(config["optim"]["lr_trans"])

    print("\n------Configurations------")
    print(f"Remote Base URL: {remote_base_url}")
    print(f"Cam IDs: {cam_ids}")
    print(f"Modalities: {modalities}")
    print(f"Split: {split}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Num Workers: {num_workers}")
    print(f"Device: {device}")
    print(f"Learning Rate (Rotation): {lr_rot}")
    print(f"Learning Rate (Translation): {lr_trans}")

    torch.backends.cudnn.benchmark = True

    # --- Dataset ---
    dataset = IPDValidationDatasetMounted(
        remote_base_url=remote_base_url,
        cam_ids=cam_ids,
        modalities=modalities,
        split=split
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    scaler = GradScaler()

    # --- Model ---
    sensory_channels = {mod: 1 for mod in modalities}
    model = FuseNetPoseModel(sensory_channels=sensory_channels, fc_width=64).to(device)
    file_path = f"./weights/fusenet_pose_{len(modalities)}y.pth"

    # --- Separate optimizers ---
    rot_params = []
    trans_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'fc.3.weight' in name:
            rot_params.append(param[:4])
            trans_params.append(param[4:])
        elif 'fc.3.bias' in name:
            rot_params.append(param[:4])
            trans_params.append(param[4:])
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": rot_params, "lr": lr_rot},
        {"params": trans_params, "lr": lr_trans},
        {"params": other_params}
    ])

    rot_losses = []
    trans_losses = []

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        for i, (x_dict, R_gt, t_gt, K) in enumerate(dataloader):
            x_dict = {modality: Variable(x).to(device) for modality, x in x_dict.items()}
            R_gt = Variable(R_gt).to(device)
            t_gt = Variable(t_gt).to(device)

            optimizer.zero_grad()
            with autocast():
                quat, trans, rot_loss, trans_loss = model(x_dict, R_gt, t_gt)
                total_loss = rot_loss + trans_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            rot_losses.append(rot_loss.item())
            trans_losses.append(trans_loss.item())

            total_loss = rot_loss.item() + trans_loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)} "
                  f"| Rot Loss: {rot_loss.item():.4f} | Trans Loss: {trans_loss.item():.4f} | Total Loss: {total_loss:.4f}")

        os.makedirs("./weights", exist_ok=True)
        torch.save(model.state_dict(), file_path)

    # Save loss curves
    with open(f"trans_loss_{len(modalities)}y.json", "w") as f:
        json.dump(trans_losses, f)

    with open(f"rot_loss_{len(modalities)}y.json", "w") as f:
        json.dump(rot_losses, f)

if __name__ == "__main__":
    main()
