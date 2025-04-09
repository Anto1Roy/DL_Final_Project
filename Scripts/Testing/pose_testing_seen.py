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
from Models.PoseEstimator.PoseEstimation import PoseEstimator
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Models.helpers import quaternion_to_matrix
from Metrics.visualization import draw_pose_axes, draw_bbox_from_pose
from Metrics.metrics import compute_add_gpu, compute_add_or_adds  # assume compute_add_gpu works on GPU


def custom_collate_fn(batch):
    # Only keep non-None samples
    return [s for s in batch if s is not None]


def move_sample_to_device(sample, device):
    X, Y = sample["X"], sample["Y"]
    # Move each view data to the device
    for view in X["views"]:
        for k in view:
            view[k] = view[k].to(device)
    X["K"] = X["K"].to(device)
    # Move CAD model tensors
    for item in X["available_cads"]:
        item["verts"] = item["verts"].to(device)
        item["faces"] = item["faces"].to(device)
    # Move ground-truth pose tensors
    for pose in Y["gt_poses"]:
        pose["R"] = pose["R"].to(device)
        pose["t"] = pose["t"].to(device)
    return sample


def evaluate_model(model, loader, device, renderer=None, save_dir="eval_outputs_seen"):
    print("[INFO] Evaluating model on seen CAD models...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_losses = []
    overall_obj_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "correct": 0})
    add_scores = []

    for idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        # Get the first valid sample in the batch (skip if batch is empty)
        sample = next((s for s in batch if s is not None), None)
        if sample is None:
            continue

        sample = move_sample_to_device(sample, device)
        X, Y = sample["X"], sample["Y"]

        # Count the number of ground truth poses per sample.
        gt_poses = Y["gt_poses"]
        for pose in gt_poses:
            overall_obj_stats[pose["obj_id"]]["gt"] += 1

        # Build a lookup table for available CAD models (seen during training).
        cad_model_lookup = {
            item["obj_id"]: {"verts": item["verts"], "faces": item["faces"]}
            for item in X["available_cads"]
        }

        # Compute the pose loss.
        with torch.no_grad():
            loss, rot_loss, trans_loss, desc_loss = model.compute_pose_loss(
                x_dict_views=X["views"],
                R_gt_list=[pose["R"] for pose in gt_poses],
                t_gt_list=[pose["t"] for pose in gt_poses],
                K_list=X["K"],
                cad_model_lookup=cad_model_lookup
            )
        all_losses.append((rot_loss.item(), trans_loss.item()))

        # Run inference to get detections and computed poses.
        with torch.no_grad():
            print(f"[INFO] Inference on sample {idx}...")
            detections, _, _ = model(
                x_dict_views=X["views"],
                K_list=X["K"],
                cad_model_lookup=cad_model_lookup
            )

        if len(detections) == 0:
            print(f"[WARNING] No detections for sample {idx}; skipping matching/visualization.")
            continue

        # -------------------------------
        # Matching predicted poses to GT poses using the Hungarian algorithm.
        n_gt = len(gt_poses)
        n_pred = len(detections)
        cost_matrix = torch.zeros((n_gt, n_pred), device=device)

        # Build cost matrix: each element is the ADD error between GT and prediction.
        for i, gt in enumerate(gt_poses):
            R_gt = gt["R"]   # (3,3)
            t_gt = gt["t"]   # (3,)
            obj_id = gt["obj_id"]
            verts = cad_model_lookup[obj_id]["verts"]
            for j, det in enumerate(detections):
                R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]  # (3,3)
                t_pred = det["trans"]  # (3,)
                cost_matrix[i, j] = compute_add_gpu(verts, R_gt, t_gt, R_pred, t_pred)

        # Solve the assignment problem (Hungarian matching)
        gt_inds, pred_inds = linear_sum_assignment(cost_matrix.cpu().numpy())
        # Create mapping: detection index → GT index.
        match_map = {pred: gt for gt, pred in zip(gt_inds, pred_inds)}

        # Define a matching threshold (adjust as needed).
        threshold = 0.5
        correct_matches = 0

        for i, j in zip(gt_inds, pred_inds):
            error = cost_matrix[i, j].item()
            print(f"[MATCHING] GT {i} matched with Pred {j} | Error: {error:.4f}")
            gt_obj = gt_poses[i]["obj_id"]
            overall_obj_stats[gt_obj]["pred"] += 1  # Count a prediction for this object.
            if error < threshold:
                correct_matches += 1
                overall_obj_stats[gt_obj]["correct"] += 1
            # Record the ADD error.
            add_scores.append(error)

        missed = n_gt - len(gt_inds)
        false_positives = n_pred - len(pred_inds)
        print(f"[MATCHING] Sample {idx} | GT: {n_gt}, Pred: {n_pred}, Correct: {correct_matches}, Missed: {missed}, FP: {false_positives}")
        
        # Visualize prediction (only for the first sample, if desired).
        if idx < 1:
            visualize_prediction_all(
                model,
                sample,
                device,
                renderer=renderer,
                detections=detections,
                match_map=match_map,
                index=idx,
                render_mode="bbox",
                save_path=os.path.join(save_dir, f"all_pred_{idx}.png"),
                render_overlay_path=os.path.join(save_dir, f"overlay_{idx}.png"),
                render_comparison_path=os.path.join(save_dir, f"comparison_{idx}.png")
            )

    losses = np.array(all_losses)
    rot_mean, trans_mean = losses.mean(axis=0)
    rot_std, trans_std = losses.std(axis=0)
    add_scores = torch.stack(add_scores).cpu().numpy() if len(add_scores) > 0 else np.array([])

    print("\n--- Evaluation Metrics ---")
    print(f"Rotation Loss     : {rot_mean:.4f} ± {rot_std:.4f}")
    print(f"Translation Loss  : {trans_mean:.4f} ± {trans_std:.4f}")
    if add_scores.size > 0:
        print(f"ADD(-S) Mean      : {add_scores.mean():.4f} ± {add_scores.std():.4f}")
    else:
        print("No ADD scores computed.")

    print("\n--- Per-Object Metrics ---")
    for obj_id, stats in overall_obj_stats.items():
        print(f"Object {obj_id:02d}: GT: {stats['gt']}, Predictions: {stats['pred']}, Correct: {stats['correct']}, Missed: {stats['gt'] - stats['correct']}")

    return losses


def visualize_prediction_all(model, sample, device, renderer,
                             detections=None, match_map=None, index=0, render_mode="bbox",
                             save_path="all_pred.png", render_overlay_path="overlay.png",
                             render_comparison_path="comparison.png"):
    """
    Visualizes ground truth and predicted poses over the image.
    Draws bounding boxes and renders mesh overlays for comparison.
    """
    X, Y = sample["X"], sample["Y"]
    cad_model_lookup = {
        item["obj_id"]: {"verts": item["verts"], "faces": item["faces"]}
        for item in X["available_cads"]
    }

    # Run inference if detections are not provided.
    if detections is None:
        with torch.no_grad():
            detections, _ = model(
                x_dict_views=X["views"],
                K_list=X["K"],
                cad_model_lookup=cad_model_lookup
            )

    # Use the first camera view as the base image.
    cam_id = 0
    mod_name = "rgb"
    view = X["views"][cam_id]
    tensor = view[mod_name][cam_id].cpu()
    img_pil = ToPILImage()(tensor)
    img_rgb = np.array(img_pil.convert("RGB"))

    # Prepare camera intrinsics.
    K_np = X["K"][0].cpu().numpy().astype(np.float32)

    # Draw bounding boxes for GT and predictions.
    vis = img_rgb.copy()
    for i, gt in enumerate(Y["gt_poses"]):
        obj_id = gt["obj_id"]
        R_gt = gt["R"].cpu().numpy()
        t_gt = gt["t"].cpu().numpy()
        vis = draw_bbox_from_pose(vis, R_gt, t_gt, K_np, label=f"GT_{obj_id}_{i}", color=(255, 0, 0))
    for i, det in enumerate(detections):
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0].cpu().numpy()
        t_pred = det["trans"].cpu().numpy()
        if match_map is not None and i in match_map:
            gt_index = match_map[i]
            label = f"Pred_{i} (matched GT_{gt_index})"
        else:
            label = f"Pred_{i} (unmatched)"
        vis = draw_bbox_from_pose(vis, R_pred, t_pred, K_np, label=label, color=(0, 255, 0))
    if save_path:
        plt.imsave(save_path, vis)

    # Create an overlay with rendered predicted meshes.
    overlay = img_rgb.copy().astype(np.float32)
    for i, det in enumerate(detections):
        if match_map is not None and i in match_map:
            gt_index = match_map[i]
            obj_id = Y["gt_poses"][gt_index]["obj_id"]
        else:
            if i < len(Y["gt_poses"]):
                obj_id = Y["gt_poses"][i]["obj_id"]
            else:
                obj_id = list(cad_model_lookup.keys())[0]
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))
        t_pred = det["trans"].unsqueeze(0)
        verts = cad_model_lookup[obj_id]["verts"].unsqueeze(0)
        faces = cad_model_lookup[obj_id]["faces"].unsqueeze(0)
        rendered_pred = renderer.render_mesh(
            verts=verts, faces=faces,
            R=R_pred,
            T=t_pred,
            K=X["K"][0],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        # Blend the rendered prediction with the overlay.
        overlay = 0.7 * rendered_pred + 0.3 * overlay
    overlay = overlay.astype(np.uint8)
    if render_overlay_path:
        plt.imsave(render_overlay_path, overlay)

    # Generate side-by-side comparisons of GT vs. predicted renders.
    gt_renders = []
    pred_renders = []
    for i, gt in enumerate(Y["gt_poses"]):
        obj_id = gt["obj_id"]
        R_gt = gt["R"].unsqueeze(0)
        t_gt = gt["t"].unsqueeze(0)
        verts = cad_model_lookup[obj_id]["verts"].unsqueeze(0)
        faces = cad_model_lookup[obj_id]["faces"].unsqueeze(0)
        rendered_gt = renderer.render_mesh(
            verts=verts, faces=faces,
            R=R_gt,
            T=t_gt,
            K=X["K"][0],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        gt_renders.append(rendered_gt)
    for i, det in enumerate(detections):
        if match_map is not None and i in match_map:
            gt_index = match_map[i]
            obj_id = Y["gt_poses"][gt_index]["obj_id"]
        else:
            if i < len(Y["gt_poses"]):
                obj_id = Y["gt_poses"][i]["obj_id"]
            else:
                obj_id = list(cad_model_lookup.keys())[0]
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))
        t_pred = det["trans"].unsqueeze(0)
        verts = cad_model_lookup[obj_id]["verts"].unsqueeze(0)
        faces = cad_model_lookup[obj_id]["faces"].unsqueeze(0)
        rendered_pred = renderer.render_mesh(
            verts=verts, faces=faces,
            R=R_pred,
            T=t_pred,
            K=X["K"][0],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        pred_renders.append(rendered_pred)
    pairs = []
    n_pairs = max(len(gt_renders), len(pred_renders))
    for i in range(n_pairs):
        gt_img = gt_renders[i] if i < len(gt_renders) else 255 * np.ones_like(img_rgb, dtype=np.uint8)
        pred_img = pred_renders[i] if i < len(pred_renders) else 255 * np.ones_like(img_rgb, dtype=np.uint8)
        # Ensure images are in RGB format.
        if gt_img.ndim == 3 and gt_img.shape[2] == 1:
            gt_img = np.repeat(gt_img, 3, axis=2)
        if pred_img.ndim == 3 and pred_img.shape[2] == 1:
            pred_img = np.repeat(pred_img, 3, axis=2)
        pair = np.concatenate([gt_img, pred_img], axis=1)
        pairs.append(pair)
    comp = np.concatenate(pairs, axis=0)
    comp = comp.astype(np.uint8)
    if render_comparison_path:
        plt.imsave(render_comparison_path, comp)


def main():
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    
    # For seen CAD models, use the training split and same allowed objects as in training.
    train_scene_ids = {f"{i:06d}" for i in range(0, 25)}
    train_obj_ids = {0, 8, 18, 19, 20}
    batch_size = 1
    num_workers = 4

    dataset = IPDDatasetMounted(
        remote_base_url,
        cam_ids,
        modalities,
        split=config["dataset"].get("train_split", "train"),  # using training split
        allowed_scene_ids=train_scene_ids,
        allowed_obj_ids=train_obj_ids
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    sensory_channels = {mod: 1 for mod in modalities}
    encoder_type = config["training"].get("encoder", "fusenet")
    fusion_type = config["training"].get("fusion", "concat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer_config = config.get("renderer", {"width": 640, "height": 480, "device": device})

    model = PoseEstimator(
        sensory_channels, renderer_config,
        encoder_type=encoder_type,
        fusion_type=fusion_type,
        n_views=len(cam_ids)
    ).to(device)

    model_path = f"weights/model_{encoder_type}_{fusion_type}.pt"
    print(f"[INFO] Loading model from {model_path}")
    if os.path.exists(model_path):
        print(f"[INFO] Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[ERROR] Model file not found at {model_path}. Exiting.")
        sys.exit(1)

    renderer = KaolinRenderer(**renderer_config)
    evaluate_model(model, test_loader, device, renderer=renderer, save_dir="eval_outputs_seen")


if __name__ == "__main__":
    main()
