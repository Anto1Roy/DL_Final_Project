# pose_test_unseen.py
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.append(os.getcwd())
from Models.KaolinRenderer import KaolinRenderer
from Models.ActualPoseEstimator.PoseEstimation import PoseEstimator
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Models.helpers import quaternion_to_matrix
from Metrics.visualization import draw_pose_axes, draw_bbox_from_pose
from Metrics.metrics import compute_add_gpu


def custom_collate_fn(batch):
    return [s for s in batch if s is not None]

def move_sample_to_device(sample, device):
    X, Y = sample["X"], sample["Y"]

    for view in X["views"]:
        for k in view:
            view[k] = view[k].to(device)
    

    X["K"] = X["K"].to(device)

    for cam in Y["gt_poses"]:
        for pose in cam:
            pose["R"] = pose["R"].to(device)
            pose["t"] = pose["t"].to(device)
            pose["obj_id"] = pose["obj_id"].to(device)

    return sample


def evaluate_model(model, loader, device, save_dir="eval_outputs_unseen", visualize_limit=3):
    print("[INFO] Evaluating model on unseen scenes...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_losses = []

    confusion = defaultdict(lambda: {"gt": 0, "pred": 0, "correct": 0, "wrong": 0})
    add_scores = []

    scene_dict = {}

    rot_errors = []
    trans_errors = []

    confusion = defaultdict(lambda: {"gt": 0, "pred": 0, "correct": 0, "wrong": 0})
    rot_errors, trans_errors = [], []

    for idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        for sample in batch:
            if sample is None:
                continue

            sample = move_sample_to_device(sample, device)
            X, Y = sample["X"], sample["Y"]

            gt_poses_all_views = Y["gt_poses"]
            first_view_gt = gt_poses_all_views[0]  # Only use first view's GT poses

            for pose in first_view_gt:
                confusion[pose["obj_id"].item()]["gt"] += 1

            with torch.no_grad():
                detections, _ = model(x_dict_views=X["views"], K_list=X["K"])

            if not detections:
                print(f"[WARNING] No detections in sample {idx}")
                continue

            n_gt = len(first_view_gt)
            n_pred = len(detections)
            cost_matrix = torch.full((n_gt, n_pred), fill_value=1e6, device=device)

            for i, gt in enumerate(first_view_gt):
                R_gt, t_gt = gt["R"], gt["t"]
                obj_id = gt["obj_id"]
                for j, det in enumerate(detections):
                    R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]
                    t_pred = det["trans"]
                    if obj_id.item() == det["obj_id"]:
                        add = compute_add_gpu(X["model_points_by_id"][int(obj_id.item())], R_gt, t_gt, R_pred, t_pred)
                        cost_matrix[i, j] = add

            gt_inds, pred_inds = linear_sum_assignment(cost_matrix.cpu().numpy())
            correct = 0

            for i, j in zip(gt_inds, pred_inds):
                gt_pose = first_view_gt[i]
                det = detections[j]

                gt_obj = gt_pose["obj_id"].item()
                pred_obj = det["obj_id"]

                R_gt, t_gt = gt_pose["R"], gt_pose["t"]
                R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]
                t_pred = det["trans"]

                confusion[pred_obj]["pred"] += 1

                if gt_obj == pred_obj and cost_matrix[i, j] < 0.5:
                    confusion[gt_obj]["correct"] += 1
                    correct += 1
                else:
                    confusion[pred_obj]["wrong"] += 1

                # Rotation error (in degrees)
                delta_R = R_gt @ R_pred.T
                trace = torch.trace(delta_R)
                trace = torch.clamp((trace - 1) / 2, -1.0, 1.0)
                angle_rad = torch.acos(trace)
                rot_errors.append(angle_rad.item() * 180.0 / np.pi)

                # Translation error (in meters)
                trans_errors.append(torch.norm(t_gt - t_pred).item())

            print(f"[MATCH] Sample {idx} — GT: {n_gt}, Pred: {n_pred}, Correct: {correct}")

        if idx >= 2:
            break

    print("\n--- Confusion Matrix ---")
    for obj_id, stats in confusion.items():
        total = stats["gt"]
        correct = stats["correct"]
        wrong = stats["wrong"]
        print(f"Obj {obj_id:02d}: GT={total}, Pred={stats['pred']}, Correct={correct}, Wrong={wrong}, Missed={total - correct}")

    if rot_errors and trans_errors:
        print("\n--- Error Statistics (matched predictions only) ---")
        print(f"Rotation Error (°): Mean={np.mean(rot_errors):.2f}, Std={np.std(rot_errors):.2f}")
        print(f"Translation Error (m): Mean={np.mean(trans_errors):.4f}, Std={np.std(trans_errors):.4f}")

    return None


def visualize_prediction_all(model, sample, device, renderer=None, detections=None, idx=0, save_path="vis.png"):
    from torchvision.transforms import ToPILImage
    X, Y = sample["X"], sample["Y"]
    cam_id = 0
    mod = "rgb"
    img = ToPILImage()(X["views"][cam_id][mod].cpu()).convert("RGB")
    img_np = np.array(img)
    K = X["K"][0].cpu().numpy()

    vis = img_np.copy()
    for i, gt in enumerate(Y["gt_poses"]):
        vis = draw_bbox_from_pose(vis, gt["R"].cpu().numpy(), gt["t"].cpu().numpy(), K, f"GT_{gt['obj_id']}", (255, 0, 0))

    for i, det in enumerate(detections):
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0].cpu().numpy()
        t_pred = det["trans"].cpu().numpy()
        label = f"Pred_{det['obj_id']}"
        vis = draw_bbox_from_pose(vis, R_pred, t_pred, K, label, (0, 255, 0))

    plt.imsave(save_path, vis)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    encoder_type = config["training"].get("encoder", "fusenet")
    fusion_type = config["training"].get("fusion", "concat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5

    val_train_scene_ids = {f"{i:06d}" for i in range(11, 14)}
    val_train_obj_ids = {0, 1, 4, 8, 10, 11, 14, 18, 19, 20}

    dataset = IPDDatasetMounted(
        remote_base_url,
        cam_ids,
        modalities,
        split=config["dataset"].get("val_split", "train"),
        allowed_scene_ids=val_train_scene_ids,
        allowed_obj_ids=val_train_obj_ids
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate_fn)
    sensory_channels = {mod: 1 for mod in modalities}
    model = PoseEstimator(sensory_channels, encoder_type=encoder_type, fusion_type=fusion_type,
                          obj_ids=val_train_obj_ids, n_views=len(cam_ids)).to(device)

    model_path = f"weights/pose_model_{encoder_type}_{fusion_type}_{len(modalities)}_{len(cam_ids)}.pt"
    print(f"[INFO] Loading model from {model_path}")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[ERROR] Model weights not found.")
        return

    evaluate_model(model, loader, device, save_dir="eval_outputs_unseen")


if __name__ == "__main__":
    main()
