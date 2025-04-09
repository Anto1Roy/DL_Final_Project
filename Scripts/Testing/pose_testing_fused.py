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

sys.path.append(os.getcwd())
from Models.KaolinRenderer import KaolinRenderer
from Models.PoseEstimator.PoseEstimation import PoseEstimator
from Classes.Dataset.IPDDataset_render import IPDDatasetMounted
from Models.helpers import quaternion_to_matrix
from Metrics.visualization import draw_pose_axes, draw_bbox_from_pose
from Metrics.metrics import compute_add_gpu, compute_add_or_adds  # assume compute_add_gpu works on GPU

from scipy.optimize import linear_sum_assignment

def custom_collate_fn(batch):
    return [s for s in batch if s is not None]

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

def evaluate_model(model, loader, device, renderer=None, save_dir="eval_outputs"):
    print("[INFO] Evaluating model...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    all_losses = []
    overall_obj_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "correct": 0})
    add_scores = []

    for idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        sample = next((s for s in batch if s is not None), None)
        if sample is None:
            continue

        sample = move_sample_to_device(sample, device)
        X, Y = sample["X"], sample["Y"]

        # Count GT per sample
        gt_poses = Y["gt_poses"]
        for pose in gt_poses:
            overall_obj_stats[pose["obj_id"]]["gt"] += 1

        # Just pass the CADs (obj_id → verts/faces) to the model
        cad_model_lookup = {
            item["obj_id"]: {
                "verts": item["verts"],
                "faces": item["faces"]
            } for item in X["available_cads"]
        }

        with torch.no_grad():
            loss, rot_loss, trans_loss, desc_loss = model.compute_pose_loss(
                x_dict_views=X["views"],
                R_gt_list=[pose["R"] for pose in Y["gt_poses"]],
                t_gt_list=[pose["t"] for pose in Y["gt_poses"]],
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
                cad_model_lookup=cad_model_lookup  # pass the computed class embeddings
            )

        # If detections list is empty, skip this sample.
        if len(detections) == 0:
            print(f"[WARNING] No detections for sample {idx}; skipping matching/visualization for this sample")
            continue

        # -------------------------------
        # Matching predicted poses to GT poses:
        n_gt = len(gt_poses)
        n_pred = len(detections)
        cost_matrix = torch.zeros((n_gt, n_pred), device=device)

        # Build cost matrix using compute_add_gpu for each GT - Prediction pair.
        for i, gt in enumerate(gt_poses):
            R_gt = gt["R"]   # (3,3), on GPU
            t_gt = gt["t"]   # (3,)
            # Assume that for each GT pose, the corresponding CAD exists in cad_model_lookup keyed by obj_id
            obj_id = gt["obj_id"]
            verts = cad_model_lookup[obj_id]["verts"]
            for j, det in enumerate(detections):
                R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]  # (3,3)
                t_pred = det["trans"]  # (3,)
                cost_matrix[i, j] = compute_add_gpu(verts, R_gt, t_gt, R_pred, t_pred)

        # Use Hungarian matching (linear_sum_assignment expects a CPU numpy array)
        gt_inds, pred_inds = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # Define a matching threshold for ADD error (adjust threshold as needed)
        threshold = 0.1
        correct_matches = 0

        for i, j in zip(gt_inds, pred_inds):
            error = cost_matrix[i, j].item()
            gt_obj = gt_poses[i]["obj_id"]
            overall_obj_stats[gt_obj]["pred"] += 1  # Count a prediction for this object
            if error < threshold:
                correct_matches += 1
                overall_obj_stats[gt_obj]["correct"] += 1
            # Record the ADD error as well for overall metric.
            add_scores.append(error)
        
        # Optionally, you can also log unmatched GT as missed and unmatched predictions as false positives.
        missed = n_gt - len(gt_inds)
        false_positives = n_pred - len(pred_inds)
        print(f"[MATCHING] Sample {idx} | GT: {n_gt}, Pred: {n_pred}, Correct: {correct_matches}, Missed: {missed}, FP: {false_positives}")
        
        # Forward the computed detections to the visualization function (no re-running of inference)
        if idx < 5:
            visualize_prediction(model, sample, device, renderer=renderer, detections=detections, index=idx,
                                   render_mode="bbox", save_path=os.path.join(save_dir, f"pred_{idx}.png"))

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
        print(f"Object {obj_id:02d}: GT: {stats['gt']}, Predictions: {stats['pred']}, Correct: {stats['correct']}, Missed: {stats['gt']-stats['correct']}")

    return losses

def visualize_prediction(model, sample, device, renderer, detections=None, index=0, render_mode="bbox",
                           save_path=None, render_overlay_path=None, render_comparison_path=None):
    X, Y = sample["X"], sample["Y"]
    cad_model_lookup = {
        item["obj_id"]: {"verts": item["verts"], "faces": item["faces"]}
        for item in X["available_cads"]
    }

    if detections is None:
        with torch.no_grad():
            detections, _ = model(
                x_dict_views=X["views"],
                K_list=X["K"],
                cad_model_lookup=cad_model_lookup
            )

    pose = Y["gt_poses"][0]
    obj_id = pose["obj_id"]

    R_gt = pose["R"].cpu().numpy()
    t_gt = pose["t"].cpu().numpy()
    R_pred = quaternion_to_matrix(detections[0]["quat"].unsqueeze(0))[0].cpu().numpy()
    t_pred = detections[0]["trans"].cpu().numpy()
    
    # Ensure K is a proper 3x3 floating point matrix for OpenCV:
    K_np = X["K"][0].cpu().numpy()
    K_np = K_np.astype(np.float32)
    
    verts = cad_model_lookup[obj_id]["verts"]
    faces = cad_model_lookup[obj_id]["faces"]

    view = X["views"][0]
    mod_name = list(view.keys())[0]
    tensor = view[mod_name][0].cpu()
    img_pil = ToPILImage()(tensor)
    img_rgb = np.array(img_pil.convert("RGB"))

    vis = draw_bbox_from_pose(img_rgb, R_gt, t_gt, K_np, label="GT", color=(255, 0, 0))
    vis = draw_bbox_from_pose(vis, R_pred, t_pred, K_np, label="Pred", color=(0, 255, 0))
    if save_path:
        plt.imsave(save_path, vis)

    rendered_pred = renderer.render_mesh(
        verts=verts, faces=faces,
        R=torch.tensor(R_pred, device=device),
        T=torch.tensor(t_pred, device=device),
        K=X["K"],  # Already safe, used by the renderer
        background="white",
        resolution=img_rgb.shape[:2]
    ).cpu().permute(1, 2, 0).numpy()

    blend = (0.7 * rendered_pred + 0.3 * img_rgb).astype(np.uint8)
    if render_overlay_path:
        plt.imsave(render_overlay_path, blend)

    rendered_gt = renderer.render_mesh(
        verts=verts, faces=faces,
        R=torch.tensor(R_gt, device=device),
        T=torch.tensor(t_gt, device=device),
        K=X["K"],  # Already safe, used by the renderer
        background="white",
        resolution=img_rgb.shape[:2]
    ).cpu().permute(1, 2, 0).numpy()

    comp = np.concatenate([rendered_gt, rendered_pred], axis=1)
    if render_comparison_path:
        plt.imsave(render_comparison_path, comp)

def visualize_prediction_all(model, sample, device, renderer,
                             detections=None, index=0, render_mode="bbox",
                             save_path="all_pred.png", render_overlay_path="overlay.png",
                             render_comparison_path="comparison.png"):
    """
    Visualizes all the objects (both GT and predicted poses) in the scene.
    
    This function:
      - Draws the bounding boxes for all ground truth poses.
      - Draws the bounding boxes for all predicted poses.
      - Creates a blended overlay of predicted meshes on the original image.
      - Generates a side-by-side comparison (per object) of the rendered GT vs. predicted meshes.
    
    Args:
        model: The pose estimator model.
        sample: Dictionary containing the sample data.
        device: The torch device.
        renderer: An instance of KaolinRenderer.
        detections: (Optional) List of predictions; if None, the model is run.
        index: Index of the sample (for labeling purposes).
        render_mode: Visualization mode (not actively used in this function).
        save_path: Path to save the overlay with drawn bounding boxes.
        render_overlay_path: Path to save the blended (mesh overlay) image.
        render_comparison_path: Path to save the side-by-side comparison image.
    """
    # Unpack the sample and build the CAD lookup table.
    X, Y = sample["X"], sample["Y"]
    cad_model_lookup = {
        item["obj_id"]: {"verts": item["verts"], "faces": item["faces"]}
        for item in X["available_cads"]
    }

    # If no detections are provided, run inference.
    if detections is None:
        with torch.no_grad():
            detections, _ = model(
                x_dict_views=X["views"],
                K_list=X["K"],
                cad_model_lookup=cad_model_lookup
            )
    
    # Get the first view to serve as our base image.
    view = X["views"][0]
    mod_name = list(view.keys())[0]
    tensor = view[mod_name][0].cpu()
    img_pil = ToPILImage()(tensor)
    img_rgb = np.array(img_pil.convert("RGB"))
    
    # Prepare the camera intrinsic: convert to 3x3 float32 numpy array.
    K_np = X["K"][0].cpu().numpy().astype(np.float32)
    
    # ---------------------------
    # Draw bounding boxes on a copy of the image.
    vis = img_rgb.copy()
    
    # Draw all Ground Truth bounding boxes.
    for i, gt in enumerate(Y["gt_poses"]):
        obj_id = gt["obj_id"]
        R_gt = gt["R"].cpu().numpy()
        t_gt = gt["t"].cpu().numpy()
        # Draw GT box (red) with a label including object id and an index.
        vis = draw_bbox_from_pose(vis, R_gt, t_gt, K_np, label=f"GT_{obj_id}_{i}", color=(255, 0, 0))
    
    # Draw all predicted bounding boxes.
    for i, det in enumerate(detections):
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0].cpu().numpy()
        t_pred = det["trans"].cpu().numpy()
        # Draw predicted box (green) with a label that uses the prediction index.
        vis = draw_bbox_from_pose(vis, R_pred, t_pred, K_np, label=f"Pred_{i}", color=(0, 255, 0))
    
    # Save the image with all bounding boxes if desired.
    if save_path:
        plt.imsave(save_path, vis)
    
    # ---------------------------
    # Create an overlay of predicted meshes on the background image.
    # Note that if multiple objects overlap, their overlay might not compose ideally.
    overlay = img_rgb.copy().astype(np.float32)
    for i, det in enumerate(detections):
        # If available, use the corresponding GT object to pick the CAD model.
        # Otherwise, fall back to using the first available CAD.
        if i < len(Y["gt_poses"]):
            gt_obj = Y["gt_poses"][i]
            obj_id = gt_obj["obj_id"]
        else:
            obj_id = list(cad_model_lookup.keys())[0]
        
        # Compute the predicted pose.
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]
        t_pred = det["trans"]
        verts = cad_model_lookup[obj_id]["verts"]
        faces = cad_model_lookup[obj_id]["faces"]
        
        rendered_pred = renderer.render_mesh(
            verts=verts,
            faces=faces,
            R=R_pred,
            T=t_pred,
            K=X["K"],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        # Blend the rendered predicted mesh with the overlay image.
        overlay = 0.7 * rendered_pred + 0.3 * overlay
    overlay = overlay.astype(np.uint8)
    
    if render_overlay_path:
        plt.imsave(render_overlay_path, overlay)
    
    # ---------------------------
    # Create a side-by-side comparison for GT and predicted meshes per object.
    gt_renders = []
    pred_renders = []
    
    # Render all GT objects.
    for i, gt in enumerate(Y["gt_poses"]):
        obj_id = gt["obj_id"]
        R_gt = gt["R"]
        t_gt = gt["t"]
        verts = cad_model_lookup[obj_id]["verts"]
        faces = cad_model_lookup[obj_id]["faces"]
        rendered_gt = renderer.render_mesh(
            verts=verts,
            faces=faces,
            R=R_gt,
            T=t_gt,
            K=X["K"],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        gt_renders.append(rendered_gt)
    
    # Render all predicted objects.
    for i, det in enumerate(detections):
        # Use the corresponding GT object if available to pick the CAD mesh.
        if i < len(Y["gt_poses"]):
            obj_id = Y["gt_poses"][i]["obj_id"]
        else:
            obj_id = list(cad_model_lookup.keys())[0]
        R_pred = quaternion_to_matrix(det["quat"].unsqueeze(0))[0]
        t_pred = det["trans"]
        verts = cad_model_lookup[obj_id]["verts"]
        faces = cad_model_lookup[obj_id]["faces"]
        rendered_pred = renderer.render_mesh(
            verts=verts,
            faces=faces,
            R=R_pred,
            T=t_pred,
            K=X["K"],
            background="white",
            resolution=img_rgb.shape[:2]
        ).cpu().permute(1, 2, 0).numpy()
        pred_renders.append(rendered_pred)
    
    # Concatenate comparison images (GT render and Pred render side by side for each object).
    pairs = []
    n_pairs = max(len(gt_renders), len(pred_renders))
    for i in range(n_pairs):
        # If a render is not available, fill with a white image.
        gt_img = gt_renders[i] if i < len(gt_renders) else 255 * np.ones_like(img_rgb, dtype=np.uint8)
        pred_img = pred_renders[i] if i < len(pred_renders) else 255 * np.ones_like(img_rgb, dtype=np.uint8)
        pair = np.concatenate([gt_img, pred_img], axis=1)
        pairs.append(pair)
    comp = np.concatenate(pairs, axis=0)
    
    if render_comparison_path:
        plt.imsave(render_comparison_path, comp)

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "Config/config_fusenet.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    remote_base_url = config["dataset"]["remote_base_url"]
    cam_ids = config["dataset"]["cam_ids"]
    modalities = config["dataset"].get("modality", ["rgb", "depth"])
    train_split = config["dataset"].get("train_split", "train_pbr")
    test_scene_ids = {f"{i:06d}" for i in range(25, 50)}
    test_obj_ids = {1, 4, 10, 11, 14}
    batch_size = 1
    num_workers = 4

    dataset = IPDDatasetMounted(remote_base_url, cam_ids, modalities,
                                split=train_split,
                                allowed_scene_ids=test_scene_ids,
                                allowed_obj_ids=test_obj_ids)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=custom_collate_fn)

    sensory_channels = {mod: 1 for mod in modalities}
    encoder_type = config["training"].get("encoder", "fusenet")
    fusion_type = config["training"].get("fusion", "concat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer_config = config.get("renderer", {"width": 640, "height": 480, "device": device})

    model = PoseEstimator(sensory_channels, renderer_config,
                          encoder_type=encoder_type,
                          fusion_type=fusion_type,
                          n_views=len(cam_ids)).to(device)

    model_path = f"weights/model_{encoder_type}_{fusion_type}.pt"
    print(f"[INFO] Loading model from {model_path}")
    if os.path.exists(model_path):
        print(f"[INFO] Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    renderer = KaolinRenderer(**renderer_config)
    evaluate_model(model, test_loader, device, renderer=renderer, save_dir="eval_outputs")

if __name__ == "__main__":
    main()
