import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Models.PoseEstimator.PoseEstimation import PoseEstimator


@torch.no_grad()
def evaluate_model(model, loader, device, seen_obj_ids):
    model.eval()
    all_metrics = {"seen": [], "unseen": []}

    for sample in tqdm(loader, desc="Evaluating"):
        if sample is None:
            continue

        sample = move_sample_to_device(sample, device)
        X, Y = sample["X"], sample["Y"]

        cad_model_lookup = {
            item["obj_id"]: {"verts": item["verts"], "faces": item["faces"]}
            for item in X["available_cads"]
        }

        R_gt_list = [pose["R"] for pose in Y["gt_poses"]]
        t_gt_list = [pose["t"] for pose in Y["gt_poses"]]

        total_loss, rot_loss, trans_loss = model.compute_pose_loss(
            x_dict_views=X["views"],
            R_gt_list=R_gt_list,
            t_gt_list=t_gt_list,
            K_list=X["K"],
            cad_model_lookup=cad_model_lookup
        )

        for pose in Y["gt_poses"]:
            obj_id = pose["obj_id"]
            key = "seen" if obj_id in seen_obj_ids else "unseen"
            all_metrics[key].append((rot_loss.item(), trans_loss.item()))

    # Average metrics
    summary = {}
    for k in all_metrics:
        if all_metrics[k]:
            rot_losses, trans_losses = zip(*all_metrics[k])
            summary[k] = {
                "rot_loss": np.mean(rot_losses),
                "trans_loss": np.mean(trans_losses),
                "count": len(rot_losses)
            }
        else:
            summary[k] = {"rot_loss": 0, "trans_loss": 0, "count": 0}
    return summary


def move_sample_to_device(sample, device):
    X, Y = sample["X"], sample["Y"]
    for view in X["views"]:
        for k in view:
            view[k] = view[k].to(device)
    X["K"] = X["K"].to(device)
    for item in X["available_cads"]:
        item["verts"] = item["verts"].to(device)
        item["faces"] = item["faces"].to(device)
    for pose in Y["gt_poses"]:
        pose["R"] = pose["R"].to(device)
        pose["t"] = pose["t"].to(device)
    return sample


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    num_workers = config["training"]["num_workers"]
    encoder_type = config["training"].get("encoder", "fusenet")
    fusion_type = config["training"].get("fusion", "concat")
    model_path = f"weights/model_{encoder_type}_{fusion_type}.pt"
    batch_size = config["training"].get("eval_batch_size", 1)

    train_scene_ids = {f"{i:06d}" for i in range(0, 25)}
    test_scene_ids = {f"{i:06d}" for i in range(25, 50)}

    train_obj_ids = {0, 8, 18, 19, 20}
    test_obj_ids = {1, 4, 10, 11, 14}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer_config = config.get("renderer", {"width": 640, "height": 480, "device": device})

    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities,
                                split=config["dataset"].get("test_split", "test"))

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             collate_fn=lambda x: x[0])  # Only one sample per batch

    sensory_channels = {mod: 1 for mod in modalities}
    model = PoseEstimator(
        sensory_channels=sensory_channels,
        renderer_config=renderer_config,
        encoder_type=encoder_type,
        fusion_type=fusion_type,
        n_views=len(cam_ids)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model from {model_path}")

    metrics = evaluate_model(model, test_loader, device, train_obj_ids)

    print("\n--- Evaluation Summary ---")
    for key in metrics:
        print(f"{key.upper()}:")
        print(f"  Rot Loss    : {metrics[key]['rot_loss']:.4f}")
        print(f"  Trans Loss  : {metrics[key]['trans_loss']:.4f}")
        print(f"  Samples     : {metrics[key]['count']}")
    print("--------------------------")


if __name__ == "__main__":
    main()
