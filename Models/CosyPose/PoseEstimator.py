import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from Models.CosyPose.MultiViewMatcher import MultiViewMatcher
from Models.CosyPose.CozyPose import CosyPoseStyleRenderMatch
from Models.CosyPose.CandidatePose import CandidatePoseModel
from Models.Encoding.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.Encoding.ResNet34 import ResNetFeatureEncoder
from Models.CosyPose.SceneRefiner import SceneRefiner
from Models.helpers import quaternion_to_matrix

def pose_distance(R1, t1, R2, t2):
    rot_diff = torch.norm(R1 - R2)
    trans_diff = torch.norm(t1 - t2)
    return rot_diff + trans_diff

class TwoStagePoseEstimator(nn.Module):
    def __init__(self, sensory_channels, renderer_config, encoder_type="resnet",
                 num_candidates=32, noise_level=0.05, conf_thresh=0.5):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh
        self.out_dim = 16

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0],
                                                out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.out_dim)

        self.det_head = CandidatePoseModel(feature_dim=self.out_dim)
        # Bbox head to predict 2D boxes from per-view feature maps
        self.bbox_head = nn.Conv2d(self.out_dim, 4, kernel_size=1)
        self.refiner = CosyPoseStyleRenderMatch(renderer_config, num_candidates=num_candidates)
        self.matcher = MultiViewMatcher(pose_threshold=0.1)
        self.global_refiner = SceneRefiner(renderer_config)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect_single_view(self, feat_map, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections_per_sample = self.det_head.decode_poses(outputs, top_k=top_k)
        filtered = [
            [d for d in dets if d.get("score", 1.0) >= self.conf_thresh]
            for dets in detections_per_sample
        ]
        return filtered, outputs

    def sample_candidates(self, detections):
        """
        For each detection in a view, sample candidate poses.
        Returns a list (one element per detection) where each element is a tensor of shape (num_candidates, 4, 4)
        representing candidate poses (in camera coordinates).
        """
        candidates_all = []
        for det in detections:
            quat = det["quat"].unsqueeze(0).expand(self.num_candidates, -1)
            trans = det["trans"].unsqueeze(0).expand(self.num_candidates, -1)
            noise_q = torch.randn_like(quat) * self.noise_level
            noise_t = torch.randn_like(trans) * self.noise_level
            noisy_q = F.normalize(quat + noise_q, dim=-1)
            noisy_t = trans + noise_t
            R = quaternion_to_matrix(noisy_q)
            poses = torch.eye(4, device=quat.device).repeat(self.num_candidates, 1, 1)
            poses[:, :3, :3] = R
            poses[:, :3, 3] = noisy_t
            candidates_all.append(poses)
        return candidates_all

    def match_gt_with_detections_across_views(self, detections_per_view, R_gt_tensor,
                                              t_gt_tensor, instance_id_list, sample_indices,
                                              feat_maps, Ks):
        matched_detections = []
        matched_gt_R, matched_gt_t, matched_ids, matched_feat, matched_K = [], [], [], [], []

        objects_by_id = defaultdict(list)
        for i in range(len(sample_indices)):
            sample_idx = sample_indices[i].item()
            obj_id = instance_id_list[i]
            objects_by_id[obj_id].append((sample_idx, R_gt_tensor[i], t_gt_tensor[i]))

        for obj_id, gt_info_list in objects_by_id.items():
            for sample_idx, R_gt, t_gt in gt_info_list:
                dets = detections_per_view[sample_idx]
                best_match, best_dist = None, float('inf')
                for det in dets:
                    quat = det["quat"]
                    t_pred = det["trans"]
                    R_pred = quaternion_to_matrix(quat.unsqueeze(0))[0]
                    dist = torch.norm(R_gt - R_pred) + torch.norm(t_gt - t_pred)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = det
                if best_match is not None:
                    matched_detections.append(best_match)
                    matched_gt_R.append(R_gt)
                    matched_gt_t.append(t_gt)
                    matched_ids.append(obj_id)
                    matched_feat.append(feat_maps[sample_idx])
                    matched_K.append(Ks[sample_idx])
        return matched_detections, matched_gt_R, matched_gt_t, matched_ids, matched_feat, matched_K

    def compute_loss(self, x_dict_views, K_list, cad_model_data, R_gt_list, t_gt_list,
                     instance_id_list, cad_model_lookup, extrinsics):
        """
        Global refinement loss based on multi-view matching.
        """
        # Stage 1: Extract features per view.
        feat_maps = [
            self.extract_single_view_features({mod: view[mod].unsqueeze(0) for mod in view})
            for view in x_dict_views
        ]
        # Stage 2: Detection (for each view, select first detection)
        detections_per_view = [self.detect_single_view(fm, top_k=25)[0][0] for fm in feat_maps]
        # Stage 3: Multi-view hypothesis matching.
        object_groups = self.matcher.match(detections_per_view, feat_maps, K_list, extrinsics)
        # Stage 4: Global scene refinement.
        total_loss, render_losses, add_s = self.global_refiner.compute_loss(
            object_groups,
            cad_model_lookup={obj_id: cad_model_data[i] for i, obj_id in enumerate(instance_id_list)},
            gt_R_tensor=torch.stack(R_gt_list) if R_gt_list else None,
            gt_t_tensor=torch.stack(t_gt_list) if t_gt_list else None,
            instance_id_list=instance_id_list
        )
        return total_loss, render_losses, add_s

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, gt_obj_ids,
                          K_list, extrinsics, bbox_gt_list=None):
        """
        Computes the auxiliary loss for the detection stage and candidate sampling,
        converting both GT and candidate poses to global (world) coordinates using the camera extrinsics.
        
        Args:
            x_dict_views: List of dicts per view {modality -> Tensor(C,H,W)}.
            R_gt_list: Nested list per view, list of GT rotation matrices (Tensor(3,3)).
            t_gt_list: Nested list per view, list of GT translation vectors (Tensor(3)).
            gt_obj_ids: Nested list per view, list of GT object IDs.
            K_list: List/Tensor of camera intrinsics (one per view).
            extrinsics: List of dicts, one per view, with keys "R_w2c" and "t_w2c" (in camera coords).
            bbox_gt_list: List of dicts (per view) containing ground-truth bounding boxes under "bbox_visib_list".
        
        Returns:
            total_loss, avg_rot, avg_trans, avg_class, avg_conf, avg_bbox
        """
        device = K_list[0].device

        # Stage A: Extract per-view features.
        feat_maps = [
            self.extract_single_view_features({mod: view[mod].unsqueeze(0) for mod in view})
            for view in x_dict_views
        ]

        # Auxiliary bounding box loss computed from each view’s feature map.
        bbox_loss = torch.tensor(0.0, device=device)
        if bbox_gt_list is not None:
            for i, (feat_map, bbox_gt_view) in enumerate(zip(feat_maps, bbox_gt_list)):
                pred_bbox_map = self.bbox_head(feat_map)  # [1, 4, H, W]
                _, _, H, W = pred_bbox_map.shape
                pred_bbox_map = pred_bbox_map[0].permute(1, 2, 0)  # [H, W, 4]
                center_h = H // 2
                center_w = W // 2
                pred_box = pred_bbox_map[center_h, center_w]  # (4,)
                if bbox_gt_view is not None and "bbox_visib_list" in bbox_gt_view and len(bbox_gt_view["bbox_visib_list"]) > 0:
                    gt_box = torch.tensor(bbox_gt_view["bbox_visib_list"][0],
                                          dtype=torch.float32, device=device)
                    bbox_loss += F.l1_loss(pred_box, gt_box)

        # Stage B: Candidate sampling from each view using CosyPose Stage 1.
        all_pred_Rs = []
        all_pred_ts = []
        for i, feat_map in enumerate(feat_maps):
            # Get detections from the current view (top_k = 5)
            detections_batch, _ = self.detect_single_view(feat_map, top_k=5)
            detections = detections_batch[0]  # list of detections for this view
            if len(detections) == 0:
                continue
            # Sample candidates for each detection
            candidates_per_detection = self.sample_candidates(detections)
            # Get extrinsics for this view
            R_w2c = extrinsics[i]["R_w2c"].to(device)  # (3,3)
            t_w2c = extrinsics[i]["t_w2c"].to(device)  # (3,)
            for cand in candidates_per_detection:
                # cand: tensor of shape (num_candidates, 4, 4) in camera coordinates.
                num_candidates = cand.shape[0]
                R_candidates = cand[:, :3, :3]
                t_candidates = cand[:, :3, 3]
                # Convert each candidate from camera to global (world) coordinates:
                # R_global = R_w2c^T @ R_candidate; t_global = R_w2c^T @ (t_candidate - t_w2c)
                R_global = torch.matmul(R_w2c.T.unsqueeze(0).expand(num_candidates, -1, -1), R_candidates)
                t_global = torch.matmul(R_w2c.T.unsqueeze(0).expand(num_candidates, -1), (t_candidates - t_w2c).unsqueeze(-1)).squeeze(-1)
                all_pred_Rs.append(R_global)
                all_pred_ts.append(t_global)

        if len(all_pred_Rs) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero, zero, zero, zero

        # Combine candidate poses from all views.
        all_pred_Rs = torch.cat(all_pred_Rs, dim=0)  # (N_pred, 3, 3)
        all_pred_ts = torch.cat(all_pred_ts, dim=0)  # (N_pred, 3)

        # Stage C: Convert GT poses from each view into global coordinates.
        gt_R_list_flat = []
        gt_t_list_flat = []
        for view_idx, (view_Rs, view_ts) in enumerate(zip(R_gt_list, t_gt_list)):
            R_w2c = extrinsics[view_idx]["R_w2c"].to(device)
            t_w2c = extrinsics[view_idx]["t_w2c"].to(device)
            for R_gt, t_gt in zip(view_Rs, view_ts):
                # Convert GT: R_global = R_w2c^T @ R_gt; t_global = R_w2c^T @ (t_gt - t_w2c)
                gt_R_list_flat.append(torch.matmul(R_w2c.T, R_gt.to(device)))
                gt_t_list_flat.append(torch.matmul(R_w2c.T, (t_gt.to(device) - t_w2c).unsqueeze(-1)).squeeze(-1))
        gt_Rs = torch.stack(gt_R_list_flat)  # (N_gt, 3, 3)
        gt_ts = torch.stack(gt_t_list_flat)    # (N_gt, 3)

        num_gt = len(gt_Rs)
        num_pred = len(all_pred_Rs)

        # Compute pairwise differences between GT and candidate poses.
        rot_diff = torch.cdist(gt_Rs.view(num_gt, -1), all_pred_Rs.view(num_pred, -1))
        trans_diff = torch.cdist(gt_ts.view(num_gt, -1), all_pred_ts.view(num_pred, -1))

        # Hungarian matching (minimizing combined rotation+translation error)
        cost_matrix = (rot_diff + trans_diff).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        losses_rot = []
        losses_trans = []
        for gt_idx, pred_idx in zip(row_ind, col_ind):
            losses_rot.append(rot_diff[gt_idx, pred_idx])
            losses_trans.append(trans_diff[gt_idx, pred_idx])
        losses_rot = torch.stack(losses_rot)
        losses_trans = torch.stack(losses_trans)
        avg_rot = losses_rot.mean()
        avg_trans = losses_trans.mean()

        # For the two-stage estimator, we do not have classification or confidence losses.
        avg_class = torch.tensor(0.0, device=device)
        avg_conf = torch.tensor(0.0, device=device)
        avg_bbox = bbox_loss / len(bbox_gt_list) if bbox_gt_list is not None and len(bbox_gt_list) > 0 else torch.tensor(0.0, device=device)

        total_loss = avg_rot + avg_trans + 0.1 * avg_bbox

        return total_loss, avg_rot, avg_trans, avg_class, avg_conf, avg_bbox

    def forward(self, x_dict_list, K_list, cad_model_lookup, extrinsics, top_k=100):
        # Stage 1: Extract features and get per-view detections.
        feat_maps = [self.extract_single_view_features(x_dict) for x_dict in x_dict_list]
        detections_per_view = [self.detect_single_view(f, top_k=top_k)[0][0] for f in feat_maps]

        print("Detections per view:")
        for i, dets in enumerate(detections_per_view):
            print(f"View {i}: {len(dets)} detections")

        # Stage 2: Multi-view hypothesis matching.
        object_groups = self.matcher.match(detections_per_view, feat_maps, K_list, extrinsics)
        print("Object groups after matching:")
        for i, group in enumerate(object_groups):
            print(f"Group {i}: {len(group)} detections")

        # Stage 3: Global scene refinement.
        refined_poses, rendered_views = self.global_refiner(object_groups, cad_model_lookup)
        print("Refined poses and rendered views:")
        for i, (pose, view) in enumerate(zip(refined_poses, rendered_views)):
            print(f"Pose {i}: {pose}")
            print(f"View {i}: {view}")

        return object_groups, refined_poses, rendered_views

    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.det_head.parameters():
            param.requires_grad = False
