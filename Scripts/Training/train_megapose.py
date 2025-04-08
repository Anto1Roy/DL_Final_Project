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

# Include current working directory
sys.path.append(os.getcwd())

# Data and utility imports
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Classes.EarlyStopping import EarlyStopping

# Use the MegaPose estimator instead of the TwoStagePoseEstimator
from Models.MegaPose.MegaPose import MegaPoseEstimator

# ------------------------------------------------------------------------------
# Define helper functions for checkpointing and collating batches
# ------------------------------------------------------------------------------

def save_checkpoint(state, filename):
    torch.save(state, filename)

def custom_collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    return batch if len(batch) > 0 else None

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

# ------------------------------------------------------------------------------
# Optionally: a function to load CAD model data.
# In a realistic setup, replace this with your own CAD models loader.
# For demonstration, we create dummy CAD models.
# ------------------------------------------------------------------------------
def load_cad_models(path=None):
    # In practice load from a file or database.
    dummy_verts = torch.rand(1000, 3)  # 1000 vertices
    dummy_faces = torch.randint(0, 1000, (300, 3))  # 300 faces
    # Create a dictionary mapping CAD model id to a dict with "verts" and "faces".
    cad_models = {
        1: {"verts": dummy_verts, "faces": dummy_faces},
        2: {"verts": dummy_verts, "faces": dummy_faces},
        8: {"verts": dummy_verts, "faces": dummy_faces},
        10: {"verts": dummy_verts, "faces": dummy_faces},
        # Add more as needed.
    }
    return cad_models

# ------------------------------------------------------------------------------
# Main training script
# ------------------------------------------------------------------------------
def main():
    # Read configuration YAML.
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_megapose.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Dataset parameters.
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
    
    renderer_config = config.get("renderer", {"width": 640, "height": 480, "fov": 60.0})
    
    # Optionally load intrinsics if provided.
    # For example, K can be a list of 3x3 matrices.
    # Here we simulate with two dummy intrinsics.
    K1 = torch.tensor([[600.0, 0, 320.0],
                       [0, 600.0, 240.0],
                       [0, 0, 1]], dtype=torch.float32)
    K2 = torch.tensor([[800.0, 0, 320.0],
                       [0, 800.0, 240.0],
                       [0, 0, 1]], dtype=torch.float32)
    K_list = [K1, K2]
    
    # Data split.
    train_scene_ids = {f"{i:06d}" for i in range(0, 25)}
    test_scene_ids = {f"{i:06d}" for i in range(25, 50)}
    
    # Define object IDs for training and testing.
    train_obj_ids = {0, 8, 18, 19, 20}
    test_obj_ids = {1, 4, 10, 11, 14}
    
    # Initialize the dataset.
    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities, split=train_split,
                                 allowed_obj_ids=train_obj_ids, allowed_scene_ids=train_scene_ids)
    
    # Create training and validation splits.
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
    
    # Define sensory channels based on modalities.
    # Adjust the number of channels as appropriate (e.g., for RGB use 3).
    # Here we assume "rgb" images have 3 channels.
    sensory_channels = {mod: (3 if mod.lower() == "rgb" else 1) for mod in modalities}
    
    # Load CAD models (from file or dummy).
    cad_models = load_cad_models(config.get("cad_models_path", None))
    
    # Instantiate the MegaPoseEstimator.
    model = MegaPoseEstimator(cad_models=cad_models,
                              encoder_type=encoder_type,
                              sensory_channels=sensory_channels,
                              out_dim=int(config["training"].get("out_dim", 64)),
                              renderer_params=renderer_config,
                              distance=float(config["renderer"].get("distance", 3.0)),
                              K=K_list).to(device)
    
    # Optional: freeze parts of the network as needed.
    if config["training"].get("freeze_candidates", False):
        model.det_head.freeze_parameters()
    if config["training"].get("freeze_refiner", False):
        for param in model.refiner.parameters():
            param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=5, verbose=True)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="checkpoint_megapose.pth")
    
    start_epoch, epoch_seed = 0, 0  # Or load checkpoint as needed.
    
    batch_train_losses = []
    batch_valid_losses = []
    
    print("\n------ Training MegaPose ------")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_sample = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if batch is None:
                continue
            
            batch_avg_loss = []
            batch_avg_render_loss = []
            batch_avg_add_s_loss = []
            
            for sample in batch:
                if sample is None:
                    continue
                
                x_dict_views = sample["views"]
                Ks = sample["K"].to(device)
                R_gt = [R.to(device) for R in sample["R_gt"]]
                t_gt = [t.to(device) for t in sample["t_gt"]]
                instance_ids = [int(i) for i in sample["instance_ids"]]
                # Note: In MegaPose, the CAD models are already loaded in the estimator.
                # You might ignore sample["cad_models"] if the estimator uses its own RenderBank.
                
                # Ensure the images are on device.
                x_dict_views = [{mod: tensor.to(device) for mod, tensor in views.items()} 
                                for views in x_dict_views]
                
                with autocast():
                    loss, render_loss, add_s_loss = model.compute_loss(
                        x_dict_views=x_dict_views,
                        K_list=Ks,
                        cad_model_data=None,  # Not needed if using internal RenderBank.
                        R_gt_list=R_gt,
                        t_gt_list=t_gt,
                        instance_id_list=instance_ids
                    )
                
                batch_avg_loss.append(loss)
                batch_avg_render_loss.append(render_loss)
                batch_avg_add_s_loss.append(add_s_loss)
            
            if len(batch_avg_loss) == 0:
                continue
            
            avg_loss = torch.stack(batch_avg_loss).mean()
            avg_render_loss = np.mean(batch_avg_render_loss)
            avg_add_s_loss = np.mean(batch_avg_add_s_loss)
            
            print(f"Batch {batch_idx} | Total Loss: {avg_loss.item():.4f} | ADD Loss: {avg_add_s_loss:.4f} | Render Loss: {avg_render_loss:.4f}")
            
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += avg_loss.item()
            train_sample += 1
            batch_train_losses.append({
                "epoch": epoch,
                "batch": batch_idx,
                "total_loss": avg_loss.item(),
                "add_loss": avg_add_s_loss,
                "render_loss": avg_render_loss
            })
        
        avg_train_loss = train_loss / max(train_sample, 1)
        print(f"Epoch {epoch+1}/{epochs} | Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation loop.
        model.eval()
        valid_loss = 0.0
        valid_sample = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                if batch is None:
                    continue
                
                batch_avg_loss = []
                batch_avg_render_loss = []
                batch_avg_add_s_loss = []
                
                for sample in batch:
                    if sample is None:
                        continue
                    
                    x_dict_views = sample["views"]
                    Ks = sample["K"].to(device)
                    R_gt = [R.to(device) for R in sample["R_gt"]]
                    t_gt = [t.to(device) for t in sample["t_gt"]]
                    instance_ids = [int(i) for i in sample["instance_ids"]]
                    
                    x_dict_views = [{mod: tensor.to(device) for mod, tensor in views.items()}
                                    for views in x_dict_views]
                    
                    loss, render_loss, add_s_loss = model.compute_loss(
                        x_dict_views=x_dict_views,
                        K_list=Ks,
                        cad_model_data=None,
                        R_gt_list=R_gt,
                        t_gt_list=t_gt,
                        instance_id_list=instance_ids
                    )
                    
                    batch_avg_loss.append(loss)
                    batch_avg_render_loss.append(render_loss)
                    batch_avg_add_s_loss.append(add_s_loss)
                
                if len(batch_avg_loss) == 0:
                    continue
                
                avg_loss = torch.stack(batch_avg_loss).mean()
                avg_render_loss = np.mean(batch_avg_render_loss)
                avg_add_s_loss = np.mean(batch_avg_add_s_loss)
                
                valid_loss += avg_loss.item()
                valid_sample += 1
                batch_valid_losses.append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_loss": avg_loss.item(),
                    "add_loss": avg_add_s_loss,
                    "render_loss": avg_render_loss
                })
                
                print(f"[VAL] Epoch {epoch} | Batch {batch_idx} | Loss: {avg_loss.item():.4f} | ADD Loss: {avg_add_s_loss:.4f}")
        
        avg_valid_loss = valid_loss / max(valid_sample, 1)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        scheduler.step(avg_valid_loss)
        early_stopping(avg_valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Save logs and checkpoint.
        with open("loss_log_megapose.json", "w") as f:
            json.dump({"train": batch_train_losses, "valid": batch_valid_losses}, f, indent=2)
        print("[INFO] Saved loss logs to loss_log_megapose.json.")
        
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch_seed": epoch_seed + 1
        }, "checkpoint_megapose.pth")
        epoch_seed += 1

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
