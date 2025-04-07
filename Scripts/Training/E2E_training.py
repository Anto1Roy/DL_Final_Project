import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Local
sys.path.append(os.getcwd())
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Classes.EarlyStopping import EarlyStopping
from Models.PoseEstimator import TwoStagePoseEstimator

def save_checkpoint(state, filename):
    torch.save(state, filename)

def custom_collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return batch

def vectorize_training_batch(batch, device):
    """
    Handles multi-view + multi-modality batches.
    Each sample in the batch contains:
        - x_dict: modality -> List[Tensor] (one per view)
        - Ks: (V, 3, 3) tensor
        - R_gt: List[Tensor]
        - t_gt: List[Tensor]
        - instance_ids: List[int]
        - cad_models: List[(verts, faces)]
    
    Returns:
        x_dict_views: modality -> Tensor of shape (B, V, C, H, W)
        Ks_views: (B, V, 3, 3)
        R_gt_all: (M, 3, 3)
        t_gt_all: (M, 3)
        instance_ids_all: List[int]
        cad_models_all: List[(verts, faces)]
        sample_indices_all: (M,) tensor mapping objects to their sample
    """
    B = len(batch)
    modalities = batch[0][0].keys()
    V = len(batch[0][0][next(iter(modalities))])  # number of views

    # Init modality-wise storage
    x_dict_views = {mod: [] for mod in modalities}
    Ks_views = []

    R_gt_all, t_gt_all = [], []
    instance_ids_all, cad_models_all, sample_indices_all = [], [], []

    for sample_idx, (x_dict, Ks, R_list, t_list, id_list, cad_list) in enumerate(batch):
        # Append per-view modalities
        for mod in modalities:
            # List[Tensor(V, C, H, W)] → append one sample of V tensors
            x_dict_views[mod].append(torch.stack(x_dict[mod]).to(device).to(torch.float16))  # (V, C, H, W)

        Ks_views.append(Ks.to(device))  # (V, 3, 3)

        for i in range(len(id_list)):
            R_gt_all.append(R_list[i].to(device))
            t_gt_all.append(t_list[i].to(device))
            instance_ids_all.append(int(id_list[i]))
            cad_models_all.append((cad_list[i][0].to(device), cad_list[i][1].to(device)))
            sample_indices_all.append(sample_idx)

    # Stack per modality to shape (B, V, C, H, W)
    x_dict_views = {mod: torch.stack(x_dict_views[mod]) for mod in modalities}
    Ks_views = torch.stack(Ks_views)  # (B, V, 3, 3)

    R_gt_all = torch.stack(R_gt_all)
    t_gt_all = torch.stack(t_gt_all)
    sample_indices_all = torch.tensor(sample_indices_all, device=device)

    return x_dict_views, Ks_views, R_gt_all, t_gt_all, instance_ids_all, cad_models_all, sample_indices_all

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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    train_split = config["dataset"].get("train_split", "train")
    val_split = config["dataset"].get("val_split", "val")
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    num_workers = int(config["training"]["num_workers"])
    device = torch.device(config["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    lr = float(config["optim"].get("lr", 1e-4))
    encoder_type = config["training"].get("encoder", "fusenet")

    patience = config["training"].get("patience", 10)
    renderer_config = config.get("renderer", {"width": 640, "height": 480, "device": device})

    torch.backends.cudnn.benchmark = True

    train_scene_ids = {f"{i:06d}" for i in range(0, 25)}
    test_scene_ids = {f"{i:06d}" for i in range(25, 50)}

    train_obj_ids = {0, 8, 18, 19, 20}
    test_obj_ids = {1, 4, 10, 11, 14}

    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities, split=train_split,
                                 allowed_obj_ids=train_obj_ids, allowed_scene_ids=train_scene_ids)

    valid_size = 0.2
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_idx = int(valid_size * num_train)
    train_idx, valid_idx = indices[split_idx:], indices[:split_idx]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, collate_fn=custom_collate_fn)

    sensory_channels = {mod: 1 for mod in modalities}
    model = TwoStagePoseEstimator(
        sensory_channels=sensory_channels,
        renderer_config=renderer_config,
        encoder_type=encoder_type
    ).to(device)

    if config["training"].get("freeze_candidates", False):
        model.freeze_candidates()

    if config["training"].get("freeze_refiner", False):
        model.freeze_refiner()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="checkpoint.pt")

    start_epoch, epoch_seed = load_checkpoint("checkpoint.pth", model, optimizer, scaler, scheduler)

    batch_train_losses = []
    batch_valid_losses = []

    print("\n------Training------")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            if batch_idx > 20:
                break

            x_dict_batch, Ks, R_gt, t_gt, instance_ids, cad_models, sample_indices = vectorize_training_batch(batch, device)

            optimizer.zero_grad()
            with autocast():
                loss, rot_loss, trans_loss, render_loss = model.compute_losses(
                    x_dict_batch, Ks, cad_models, R_gt, t_gt, instance_ids, sample_indices
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batch_train_losses.append({
                "epoch": epoch,
                "batch": batch_idx,
                "total_loss": loss.item(),
                "rot_loss": rot_loss.item(),
                "trans_loss": trans_loss.item(),
                "render_loss": render_loss.item() if render_loss is not None else 0
            })

            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | Rot Loss: {rot_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                if batch is None:
                    continue

                if batch_idx > 10:
                    break

                x_dict_batch, Ks, R_gt, t_gt, instance_ids, cad_models, sample_indices = vectorize_training_batch(batch, device)

                loss, rot_loss, trans_loss, render_loss = model.compute_losses(
                    x_dict_batch, Ks, cad_models, R_gt, t_gt, instance_ids, sample_indices
                )

                valid_loss += loss.item()
                batch_valid_losses.append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_loss": loss.item(),
                    "rot_loss": rot_loss.item(),
                    "trans_loss": trans_loss.item(),
                    "render_loss": render_loss.item() if render_loss is not None else 0
                })

                print(f"[VAL] Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | Rot Loss: {rot_loss.item():.4f}")

        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        scheduler.step(avg_valid_loss)
        early_stopping(avg_valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        with open("loss_log.json", "w") as f:
            json.dump({
                "train": batch_train_losses,
                "valid": batch_valid_losses
            }, f, indent=2)
        print("[INFO] Saved batch loss logs to loss_log.json.")

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch_seed": epoch_seed + 1
        }, "checkpoint.pth")
        epoch_seed += 1

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()