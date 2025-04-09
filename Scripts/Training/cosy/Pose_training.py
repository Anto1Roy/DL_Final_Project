# Pose_Training.py
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.getcwd())
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Models.PoseEstimator import TwoStagePoseEstimator

def save_checkpoint(model, optimizer, scaler, epoch, batch_idx, path="checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict()
    }, path)
    print(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        print(f"[INFO] Loaded checkpoint from {path}")
        return checkpoint["epoch"], checkpoint.get("batch_idx", 0)
    else:
        print(f"[INFO] No checkpoint found at {path}. Starting from scratch.")
        return 0, 0


def custom_collate_fn(batch):
    return [sample for sample in batch if sample is not None]

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
    model.freeze_refiner()  # Only train stage 1

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scaler = GradScaler()

    start_epoch, start_batch_idx = load_checkpoint(model, optimizer, scaler, path=f"weights/checkpoint_pose.pth")

    print("\n------ Stage 1 Pose Training ------")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        total_samples = 0

        batch_trans_loss = []
        batch_rot_loss = []
        index = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if batch is None:
                continue

            index += 1

            optimizer.zero_grad()

            trans_loss_avg = 0.0
            rot_loss_avg = 0.0

            batch_avg_loss = []

            for sample in batch:
                x_dict_views = sample["views"]
                Ks = sample["K"].to(device)
                R_gt = [R.to(device) for R in sample["R_gt"]]
                t_gt = [t.to(device) for t in sample["t_gt"]]

                x_dict_views = [
                    {mod: img.to(device) for mod, img in view.items()}
                    for view in x_dict_views
                ]
            

                with autocast():
                    total_loss, rot_loss, trans_loss = model.compute_pose_loss(
                        x_dict_views, R_gt, t_gt, Ks
                    )

                    trans_loss_avg += trans_loss.item()
                    rot_loss_avg += rot_loss.item()
                    
                    batch_avg_loss.append(total_loss)
                    total_samples += 1

                    batch_rot_loss.append(rot_loss.item())
                    batch_trans_loss.append(trans_loss.item())

            avg_loss = torch.stack(batch_avg_loss).mean()  # keeps gradient + stays on GPU
            total_rot_loss += rot_loss_avg
            total_trans_loss += trans_loss_avg

            print(f"Batch {index} | Trans Loss {trans_loss_avg} | Rot Loss {rot_loss_avg}")

            batch_trans_loss.append(trans_loss_avg/len(batch))
            batch_rot_loss.append(rot_loss_avg/len(batch))

            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scheduler.step(avg_loss)
            scaler.update()

            if index % 1000 == 0:
                save_checkpoint(model, optimizer, scaler, epoch, index, path=f"weights/checkpoint_pose.pth")

        avg_rot_loss = total_rot_loss / total_samples
        avg_trans_loss = total_trans_loss / total_samples
        print(f"Epoch {epoch+1}: Avg Rot Loss = {avg_rot_loss:.4f}, Avg Trans Loss = {avg_trans_loss:.4f}")

        torch.save(model.state_dict(), f"weights/stage1_encoder_epoch{epoch+1}.pth")
        print(f"[INFO] Saved model weights to weights/stage1_encoder_epoch{epoch+1}.pth")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
