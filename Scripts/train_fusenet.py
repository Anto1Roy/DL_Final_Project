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

# --- Local Import ---
sys.path.append(os.getcwd())

from Models.model_fusenet_2mod import FuseNetPoseModel
from IPDDataset import IPDValidationDataset

def main():

    # --- Config ---

    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet_2.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = config["dataset"]["root_dir"]
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
    print(f"Data Root: {data_root}")
    print(f"Cam IDs: {cam_ids}")
    print(f"Modalities: {modalities}")
    print(f"Split: {split}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Num Workers: {num_workers}")
    print(f"Device: {device}")
    print(f"Learning Rate (Rotation): {lr_rot}")
    print(f"Learning Rate (Translation): {lr_trans}")

    # --- Dataset ---
    dataset = IPDValidationDataset(root_dir=data_root, cam_ids=cam_ids, modalities=modalities, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # --- Model ---
    in_channels = len(modalities)
    # model = nn.DataParallel(FuseNetPoseModel(in_channels=in_channels)).to(device)
    model = FuseNetPoseModel(in_channels=in_channels).to(device)

    if os.path.exists("./weights/fusenet_pose.pkl"):
        model.load_state_dict(torch.load(f"./weights/fusenet_pose_{len(modalities)}x.pkl"))
        print("\n--------weights restored--------\n")
    else:
        print("\n--------new weights created--------\n")

    # --- Separate optimizers ---
    rot_params = []
    trans_params = []

    # for name, param in model.named_parameters():
    #     if 'fc.6.weight' in name:
    #         rot_params.append(param[:4, :])     # first 4 output rows
    #         trans_params.append(param[4:, :])   # last 3 output rows
    #     elif 'fc.6.bias' in name:
    #         rot_params.append(param[:4])        # first 4 elements
    #         trans_params.append(param[4:])      # last 3 elements
    #     else:
    #         rot_params.append(param)
    #         trans_params.append(param)

    # --- Separate optimizers ---
    rot_optimizer = torch.optim.Adam(model.parameters(), lr=lr_rot)
    trans_optimizer = torch.optim.Adam(model.parameters(), lr=lr_trans)

    rot_losses = []
    trans_losses = []

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        for i, (x, R_gt, t_gt, K) in enumerate(dataloader):
            if i > 80:
                break
            x = Variable(x).to(device)
            R_gt = Variable(R_gt).to(device)
            t_gt = Variable(t_gt).to(device)

            rot_optimizer.zero_grad()
            print("here")
            quat, trans, rot_loss, trans_loss = model(x, R_gt, t_gt)

            # Inside training loop:
            rot_losses.append(rot_loss.item())
            trans_losses.append(trans_loss.item())

            rot_loss.backward(retain_graph=True)
            trans_loss.backward()

            rot_optimizer.step()
            trans_optimizer.step()

            total_loss = rot_loss.item() + trans_loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(dataloader)} | Rot Loss: {rot_loss.item():.4f} | Trans Loss: {trans_loss.item():.4f} | Total Loss: {total_loss:.4f}")

        # Save weights checkpoint
        os.makedirs("./weights", exist_ok=True)
        torch.save(model.state_dict(), f"./weights/fusenet_pose_{len(modalities)}x.pkl")

    # save losses
    with open(f"trans_loss_{len(modalities)}x", "w") as f:
        json.dump(trans_losses, f)    

    with open(f"rot_loss_{len(modalities)}x", "w") as f:
        json.dump(rot_losses, f)    

if __name__ == "__main__":
    main()
