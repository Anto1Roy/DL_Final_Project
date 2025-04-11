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
                 num_candidates=32, noise_level=0.05, conf_thresh=0.5
                 ):
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
        self.refiner = CosyPoseStyleRenderMatch(renderer_config)
        self.matcher = MultiViewMatcher(pose_threshold=0.1)
        self.global_refiner = SceneRefiner(renderer_config)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect_single_view(self, feat_map, extrinsics, top_k=15):
        outputs = self.det_head.forward(feat_map, extrinsics=extrinsics)
        detections_per_sample = self.det_head.decode_poses(outputs, top_k=top_k)
        filtered = [
            [d for d in dets if d.get("score", 1.0) >= self.conf_thresh]
            for dets in detections_per_sample
        ]
        return filtered, outputs

    def sample_candidates(self, detections):
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

    def compute_loss(self, x_dict_views, K_list, cad_model_lookup, R_gt_list, t_gt_list,
                 instance_id_list, extrinsics):
        device = K_list[0].device

        # -----------------------------
        # Stage 1: Feature extraction.
        # -----------------------------
        feat_maps = [
            self.extract_single_view_features({mod: view[mod].unsqueeze(0) for mod in view})
            for view in x_dict_views
        ]
        
        # -----------------------------
        # Stage 2: Detection on each view.
        # -----------------------------
        detections_per_view = [self.detect_single_view(feat_map, extrinsics[i], top_k=25)[0][0] for (i, feat_map) in enumerate(feat_maps)]
        
        # -----------------------------
        # Stage 3: Multi-view detection merging.
        # -----------------------------
        merged_detections = self.matcher.match(detections_per_view, K_list, extrinsics)
        
        # -----------------------------
        # Stage 4: Build GT view dictionaries.
        # -----------------------------
        gt_views = []
        for view_idx, (R_gts, t_gts) in enumerate(zip(R_gt_list, t_gt_list)):
            if len(R_gts) == 0:
                continue  
            gt_view = {
                'view_id': view_idx,
                'K': K_list[view_idx],
                'cam_extrinsics': extrinsics[view_idx],
                'instance_id': instance_id_list[view_idx],
                'rendered_gt': torch.zeros((1, 1, self.global_refiner.renderer.height,
                                            self.global_refiner.renderer.width), device=device)
            }
            gt_views.append(gt_view)
            
        # -----------------------------
        # Stage 5: Global scene refinement.
        # -----------------------------
        refined = self.global_refiner.refine_scene(merged_detections, gt_views, cad_model_lookup)
        
        # -----------------------------
        # Stage 6: Compute a supervision loss.
        # -----------------------------
        total_loss = torch.tensor(0.0, device=device)
        num_losses = 0
        for gt_view in gt_views:
            view_idx = gt_view["view_id"]
            if len(R_gt_list[view_idx]) == 0:
                continue
            gt_R = R_gt_list[view_idx][0].to(device)
            gt_t = t_gt_list[view_idx][0].to(device)
            for r in refined:
                if r["instance_id"] == gt_view["instance_id"]:
                    R_refined = quaternion_to_matrix(r["quat"].unsqueeze(0))[0]
                    t_refined = r["trans"]
                    total_loss += pose_distance(R_refined, t_refined, gt_R, gt_t)
                    num_losses += 1
                    break
                    
        if num_losses > 0:
            total_loss = total_loss / num_losses

        render_losses = torch.tensor(0.0, device=device)
        add_s = torch.tensor(0.0, device=device)
        
        return total_loss, render_losses, add_s


    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, gt_obj_ids,
                          K_list, extrinsics, bbox_gt_list=None):
        device = K_list[0].device

        feat_maps = [
            self.extract_single_view_features({mod: view[mod] for mod in view})
            for view in x_dict_views
        ]

        all_pred_Rs = []
        all_pred_ts = []
        for i, feat_map in enumerate(feat_maps):
            detections_batch, _ = self.detect_single_view(feat_map, extrinsics[i], top_k=25)
            detections = detections_batch[0]  # list of detections for this view
            if len(detections) == 0:
                continue

            # Get extrinsics from the current view (assumed shape: R_w2c: (3,3), t_w2c: (3,))
            candidates_per_detection = self.sample_candidates(detections)
            R_w2c = extrinsics[i]["R_w2c"].to(device)  # (3,3)
            t_w2c = extrinsics[i]["t_w2c"].to(device)  # (3,)

            for cand in candidates_per_detection:
                num_candidates = cand.shape[0]
                R_candidates = cand[:, :3, :3]  # (num_candidates, 3, 3)
                t_candidates = cand[:, :3, 3]   # (num_candidates, 3)

                # Compute the transpose of R_w2c.
                R_w2c_T = R_w2c.transpose(0, 1)  # (3, 3)
                R_w2c_exp = R_w2c_T.unsqueeze(0).expand(num_candidates, R_w2c_T.size(0), R_w2c_T.size(1))
                
                R_global = torch.matmul(R_w2c_exp, R_candidates)  # (num_candidates, 3, 3)
                
                t_diff = (t_candidates - t_w2c).unsqueeze(-1)  # (num_candidates, 3, 1)
                t_global = torch.matmul(R_w2c_exp, t_diff).squeeze(-1)  # (num_candidates, 3)
                
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
                gt_R_list_flat.append(torch.matmul(R_w2c.T, R_gt.to(device)))
                gt_t_list_flat.append(torch.matmul(R_w2c.T, (t_gt.to(device) - t_w2c).unsqueeze(-1)).squeeze(-1))

        if len(gt_R_list_flat) == 0 or len(gt_t_list_flat) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero, zero, zero
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

        total_loss = avg_rot + avg_trans + avg_class + avg_conf

        return total_loss, avg_rot, avg_trans, avg_class, avg_conf

    def forward(self, x_dict_list, K_list, cad_model_lookup, extrinsics, top_k=100):
        feat_maps = [self.extract_single_view_features(x_dict) for x_dict in x_dict_list]
        detections_per_view = [self.detect_single_view(f, top_k=top_k)[0][0] for f in feat_maps]

        print("Detections per view:")
        for i, dets in enumerate(detections_per_view):
            print(f"View {i}: {len(dets)} detections")

        object_groups = self.matcher.match(
            detections_per_view=detections_per_view,
            feat_maps=feat_maps,
            K_list=K_list,
            extrinsics=extrinsics
        )

    

        print("Object groups after matching:")
        for i, group in enumerate(object_groups):
            print(f"Group {i}: {len(group)} detections")

        refined_poses = self.global_refiner.refine_scene(object_groups, extrinsics, cad_model_lookup)

        print("Refined poses:")
        for i, pose in enumerate(refined_poses):
            print(f"Pose {i}: {pose}")

        return object_groups, 

    def forward_pose(self, x_dict_views, K_list, extrinsics, top_k=25):
        device = K_list[0].device

        # -----------------------------
        # Stage 1: Extract per-view features.
        # -----------------------------
        feat_maps = [
            self.extract_single_view_features({mod: view[mod].unsqueeze(0) for mod in view})
            for view in x_dict_views
        ]

        # -----------------------------
        # Stage 2: Run detection to get candidate detections per view.
        # -----------------------------
        detections_per_view = []
        for feat_map in feat_maps:
            det_batch, _ = self.detect_single_view(feat_map, top_k=top_k)
            # det_batch is a list (one per sample); using the first sample only.
            detections_per_view.append(det_batch[0])

        # -----------------------------
        # Stage 3: Candidate Pose Sampling and Global Conversion.
        # -----------------------------
        all_pred_Rs = []
        all_pred_ts = []
        candidate_info = []  # List to store additional information about each candidate.
        for i, detections in enumerate(detections_per_view):
            if len(detections) == 0:
                continue  # If no detection for view i, skip.
            # Get candidate poses (each of shape: (num_candidates, 4, 4) in camera frame)
            candidates_per_detection = self.sample_candidates(detections)
            R_w2c = extrinsics[i]["R_w2c"].to(device)  # (3, 3)
            t_w2c = extrinsics[i]["t_w2c"].to(device)  # (3,)

            # For every candidate from the sampled detections, convert to global coordinates.
            for det_idx, cand in enumerate(candidates_per_detection):
                num_candidates = cand.shape[0]
                R_candidates = cand[:, :3, :3]  # Candidate rotations in camera frame.
                t_candidates = cand[:, :3, 3]   # Candidate translations in camera frame.

                # Convert from camera to global (world) coordinates:
                R_global = torch.matmul(R_w2c.T.unsqueeze(0).expand(num_candidates, -1, -1),
                                        R_candidates)
                t_global = torch.matmul(R_w2c.T.unsqueeze(0).expand(num_candidates, -1),
                                        (t_candidates - t_w2c).unsqueeze(-1)).squeeze(-1)

                all_pred_Rs.append(R_global)
                all_pred_ts.append(t_global)

                for cand_idx in range(num_candidates):
                    candidate_info.append({
                        "view_idx": i,
                        "detection_idx": det_idx,
                        "candidate_idx": cand_idx,
                        "R_global": R_global[cand_idx],
                        "t_global": t_global[cand_idx],
                    })

        if len(all_pred_Rs) == 0:
            # If no candidates were produced, return None (or you could return dummy tensors)
            return None, None, []

        # Concatenate all candidates along the first dimension.
        all_pred_Rs = torch.cat(all_pred_Rs, dim=0)  # Shape: (N_pred, 3, 3)
        all_pred_ts = torch.cat(all_pred_ts, dim=0)  # Shape: (N_pred, 3)

        return all_pred_Rs, all_pred_ts, candidate_info

