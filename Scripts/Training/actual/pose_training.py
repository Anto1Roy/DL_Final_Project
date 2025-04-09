# pose_training_fused.py
import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.append(os.getcwd())
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Classes.EarlyStopping import EarlyStopping
from Models.PoseEstimatorSeen.PoseEstimation import PoseEstimator


def save_checkpoint(model, optimizer, scaler, epoch, batch_idx, path="checkpoint_pose_fused.pt"):
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict()
    }, path)
    print(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, path="weights/checkpoint_pose_fused.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        print(f"[INFO] Loaded checkpoint from {path}")
        return checkpoint["epoch"], checkpoint.get("batch_idx", 0)
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")
        return 0, 0


def custom_collate_fn(batch):
    return [s for s in batch if s is not None]

def move_sample_to_device(sample, device):
    X, Y = sample["X"], sample["Y"]

    for view in X["views"]:
        for k in view:
            view[k] = view[k].to(device)

    X["K"] = X["K"].to(device)

    for pose in Y["gt_poses"]:
        pose["R"] = pose["R"].to(device)
        pose["t"] = pose["t"].to(device)
        pose["obj_id"] = pose["obj_id"].to(device)

    for obj_id in X["model_points_by_id"]:
        X["model_points_by_id"][obj_id] = X["model_points_by_id"][obj_id].to(device)

    return sample


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    num_workers = config["training"]["num_workers"]
    lr = float(config["optim"].get("lr", 1e-4))
    encoder_type = config["training"].get("encoder", "fusenet")
    fusion_type = config["training"].get("fusion", "concat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patience = config["training"].get("patience", 10)
    checkpoint_index = config["training"].get("checkpoint_index", 50)
    val_every = float(config["training"].get("val_every", 50))
    checkpoint_file = f"pose_checkpoint_{encoder_type}_{fusion_type}.pt"
    model_path = f"weights/pose_model_{encoder_type}_{fusion_type}.pt"

    train_scene_ids = {f"{i:06d}" for i in range(0, 25)}
    train_obj_ids = {0, 8, 18, 19, 20}

    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities,
                                 split=config["dataset"].get("train_split", "train"),
                                 allowed_scene_ids=train_scene_ids,
                                 allowed_obj_ids=train_obj_ids)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    val_split = int(0.2 * len(dataset))
    train_idx, val_idx = indices[val_split:], indices[:val_split]

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=num_workers, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(val_idx),
                              num_workers=num_workers, collate_fn=custom_collate_fn)

    sensory_channels = {mod: 1 for mod in modalities}
    model = PoseEstimator(
        sensory_channels=sensory_channels,
        encoder_type=encoder_type,
        fusion_type=fusion_type,
        n_views=len(cam_ids),
        num_classes=len(train_obj_ids),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    scaler = GradScaler()

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)


    start_epoch, _ = load_checkpoint(model, optimizer, scaler, path=checkpoint_file)

    for epoch in range(start_epoch, epochs):
        if early_stopping.early_stop:
            print("[EARLY STOPPING TRIGGERED] Skipping Epochs.")
            break
        model.train()
        total_rot_loss, total_trans_loss, total_samples = 0, 0, 0
        index = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if not batch:
                continue

            index += 1

            optimizer.zero_grad()
            losses = []
            trans_loss_avg = 0.0
            rot_loss_avg = 0.0
            class_loss_avg = 0.0
            conf_loss_avg = 0.0
            # reproj_loss_avg = 0.0
            sample_avg = 0

            for sample in batch:
                sample = move_sample_to_device(sample, device)

                X, Y = sample["X"], sample["Y"]

                with autocast():
                    total_loss, avg_rot, avg_trans, avg_class, avg_conf  = model.compute_pose_loss(
                        x_dict_views=X["views"],
                        R_gt_list=[pose["R"] for pose in Y["gt_poses"]],
                        t_gt_list=[pose["t"] for pose in Y["gt_poses"]],
                        gt_obj_ids=[pose["obj_id"] for pose in Y["gt_poses"]],  
                        model_points_by_id=X["model_points_by_id"],
                        K_list=X["K"],
                    )

                losses.append(total_loss)
                rot_loss_avg += avg_rot.item()
                trans_loss_avg += avg_trans.item()
                class_loss_avg += avg_class.item()
                conf_loss_avg += avg_conf.item()
                # reproj_loss_avg += avg_reproj.item()
                sample_avg += 1

            mean_loss = torch.stack(losses).mean()
            scaler.scale(mean_loss).backward()
            scaler.step(optimizer)
            scheduler.step(mean_loss)
            scaler.update()

            print(f"[INFO] Batch {index} Loss: {mean_loss.item():.4f} | Trans: {trans_loss_avg/sample_avg:.4f} | Rot: {rot_loss_avg/sample_avg:.4f}")
            print(f"[INFO] Class Loss: {class_loss_avg/sample_avg:.4f} | Conf Loss: {conf_loss_avg/sample_avg:.4f}") #| Reproj Loss: {reproj_loss_avg/sample_avg:.4f}")

            if index % checkpoint_index == 0:
                save_checkpoint(model, optimizer, scaler, epoch, index, path=checkpoint_file)

            if index % val_every == 0:
                early_stopping(mean_loss.item(), model)

            if early_stopping.early_stop:
                print("[EARLY STOPPING TRIGGERED] Stopping training early based on training loss.")
                break

        total_rot_loss += rot_loss_avg
        total_trans_loss += trans_loss_avg
        total_samples += sample_avg

        print(f"Epoch {epoch+1} | Rot Loss: {total_rot_loss/total_samples:.4f}, "
              f"Trans Loss: {total_trans_loss/total_samples:.4f}")

        save_checkpoint(model, optimizer, scaler, epoch, 0, path=checkpoint_file)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
