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
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler

# --- Local Import ---
sys.path.append(os.getcwd())

from Models.model_fusenet_4mod import FuseNetPoseModel
from IPDDatasetStreaming import StreamingIPDIterableDataset

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet_2.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    split = config["dataset"].get("split", "val")
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    num_workers = int(config["training"]["num_workers"])
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    max_samples = config.get("streaming", {}).get("max_samples", 100000)
    skip = config.get("streaming", {}).get("skip", 0)

    lr_rot = float(config["optim"]["lr_rot"])
    lr_trans = float(config["optim"]["lr_trans"])

    checkpoint_path = f"./weights/fusenet_pose_{len(modalities)}z.pth"
    state_path = "training_state.json"
    start_epoch = 0
    seen_samples = 0

    # Load training state if exists
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
            start_epoch = state.get("epoch", 0)
            seen_samples = state.get("seen_samples", 0)
            print(f"🔁 Resuming from epoch {start_epoch}, sample {seen_samples}")

    stream_split = f"{split}_pbr"
    stream_data = load_dataset("bop-benchmark/ipd", split=stream_split, streaming=True)

    dataset = StreamingIPDIterableDataset(
        stream_data=stream_data,
        modalities=modalities,
        skip=seen_samples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Initialize model
    sensory_channels = {mod: 1 for mod in modalities}
    model = FuseNetPoseModel(sensory_channels=sensory_channels, fc_width=64).to(device)

    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    # Separate optimizers
    rot_params, trans_params, other_params = [], [], []
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

    scaler = GradScaler()
    rot_losses, trans_losses = [], []
    samples_seen = seen_samples

    for epoch in range(start_epoch, epochs):
        model.train()
        for i, (x_dict, R_gt, t_gt, K) in enumerate(dataloader):
            x_dict = {modality: Variable(x).to(device) for modality, x in x_dict.items()}
            R_gt, t_gt = Variable(R_gt).to(device), Variable(t_gt).to(device)

            optimizer.zero_grad()
            with autocast():
                quat, trans, rot_loss, trans_loss = model(x_dict, R_gt, t_gt)
                total_loss = rot_loss + trans_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            rot_losses.append(rot_loss.item())
            trans_losses.append(trans_loss.item())

            samples_seen += R_gt.size(0)

            print(f"Epoch {epoch+1}/{epochs} | Batch {i+1} | Rot Loss: {rot_loss.item():.4f} | Trans Loss: {trans_loss.item():.4f}")

        torch.save(model.state_dict(), checkpoint_path)
        with open(state_path, "w") as f:
            json.dump({"epoch": epoch + 1, "seen_samples": samples_seen}, f)

    with open(f"trans_loss_{len(modalities)}z.json", "w") as f:
        json.dump(trans_losses, f)
    with open(f"rot_loss_{len(modalities)}z.json", "w") as f:
        json.dump(rot_losses, f)

if __name__ == "__main__":
    main()
