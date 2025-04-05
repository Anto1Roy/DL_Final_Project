# --- Imports ---
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import os
import yaml
import sys
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# --- Local Import ---
sys.path.append(os.getcwd())
from Models.PoseEstimator import CosyPoseStyleRenderMatch  
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Classes.EarlyStopping import EarlyStopping

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer, scaler, scheduler):
    if os.path.exists(filepath):
        print(f"\nLoading checkpoint from {filepath}\n")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        epoch_seed = checkpoint.get("epoch_seed", start_epoch)
        return start_epoch, epoch_seed
    return 0, 0

def main():
    # --- Config ---
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet_2.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    train_split = config["dataset"].get("train_split", "train_pbr")
    val_split = config["dataset"].get("val_split", "val")
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    num_workers = int(config["training"]["num_workers"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu") 

    lr_rot = float(config["optim"]["lr_rot"])
    lr_trans = float(config["optim"]["lr_trans"])
    patience = config["training"].get("patience", 10)

    renderer_config = config.get("renderer", {"image_size": 256, "device": device})

    print("\n------Configurations------")
    print(f"Remote Base URL: {remote_base_url}")
    print(f"Cam IDs: {cam_ids}")
    print(f"Modalities: {modalities}")
    print(f"Train Split: {train_split}")
    print(f"Validation Split: {val_split}")
    print(f"Renderer Config: {renderer_config}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Num Workers: {num_workers}")
    print(f"Device: {device}")
    print(f"Learning Rate (Rotation): {lr_rot}")
    print(f"Learning Rate (Translation): {lr_trans}")

    torch.backends.cudnn.benchmark = True

    # --- Dataset ---
    dataset = IPDDatasetMounted(
        remote_base_url=remote_base_url,
        cam_ids=cam_ids,
        modalities=modalities,
        split=train_split,
        models_dir=os.path.join(remote_base_url, "models")
    )

    # --- Validation Split ---
    valid_size = 0.2
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_idx = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split_idx:], indices[:split_idx]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    # --- Model ---
    sensory_channels = {mod: 1 for mod in modalities}
    model = CosyPoseStyleRenderMatch(sensory_channels, renderer_config).to(device)
    file_path = f"./weights/rendermatch_pose_{len(modalities)}y.pth"
    checkpoint_path = f"./checkpoints/rendermatch_checkpoint_{len(modalities)}y.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=min(lr_rot, lr_trans))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="checkpoint.pt")

    rot_losses = []
    trans_losses = []
    render_losses = []
    rot_losses_val = []
    trans_losses_val = []
    render_losses_val = []
    avg_train_losses = []
    avg_valid_losses = []

    start_epoch, epoch_seed = load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler)

    print("\n------Training------")

    for epoch in range(start_epoch, epochs):
        model.train()
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        train_loss = 0.0

        for batch in train_loader:
            if batch is None:
                continue
            (
                x_dict,
            R_gt_list,
            t_gt_list,
            K,
            cad_model_data_list,
            candidate_pose_list,
            instance_id_list,
            ) = batch

            x_dict = {
                modality: Variable(x).to(device)
                for modality, x in x_dict.items()
            }
            R_gt_list = [Variable(R).to(device) for R in R_gt_list]
            t_gt_list = [Variable(t).to(device) for t in t_gt_list]
            candidate_pose_list = [p.to(device) for p in candidate_pose_list]
            cad_model_data_list = [
                (verts.to(device), faces.to(device)) for (verts, faces) in cad_model_data_list
            ]

            optimizer.zero_grad()
            with autocast():
                loss, rot_loss, trans_loss, render_loss = model.compute_losses(
                    x_dict, cad_model_data_list, candidate_pose_list, R_gt_list, t_gt_list, instance_id_list, K=K
                )

            rot_losses.append(rot_loss.item())
            trans_losses.append(trans_loss.item())
            render_losses.append(render_loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None:
                    continue
                (
                    x_dict,
                    R_gt_list,
                    t_gt_list,
                    K,
                    cad_model_data_list,
                    candidate_pose_list,
                    instance_id_list,
                ) = batch

                x_dict = {
                    modality: x.to(device) for modality, x in x_dict.items()
                }
                R_gt_list = [R.to(device) for R in R_gt_list]
                t_gt_list = [t.to(device) for t in t_gt_list]
                candidate_pose_list = [p.to(device) for p in candidate_pose_list]
                cad_model_data_list = [
                    (verts.to(device), faces.to(device)) for (verts, faces) in cad_model_data_list
                ]

                loss, rot_loss, trans_loss, render_loss = model.compute_losses(
                    x_dict, cad_model_data_list, candidate_pose_list, R_gt_list, t_gt_list, instance_id_list, K=K
                )

                rot_losses_val.append(rot_loss)
                trans_losses_val.append(trans_loss)
                render_losses_val.append(render_loss)
                valid_loss += loss.item()

                avg_valid_loss = valid_loss / len(valid_loader)
                avg_valid_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

        scheduler.step(avg_valid_loss)
        early_stopping(avg_valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch_seed": epoch_seed + 1
        }, checkpoint_path)

        epoch_seed += 1

    # Save completion flag
    with open("./.done_rendermatch.flag", "w") as f:
        f.write("done\n")

    with open(f"trans_loss_{len(modalities)}y.json", "w") as f:
        json.dump(avg_valid_losses, f)

    with open(f"rot_loss_{len(modalities)}y.json", "w") as f:
        json.dump(avg_train_losses, f)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()