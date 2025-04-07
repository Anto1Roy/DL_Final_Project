import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder
from Models.ObjectDetection.GridObjectDetector import GridObjectDetector
from Models.CandidatePose import CandidatePoseModel
from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.helpers import quaternion_to_matrix

def pose_distance(R1, t1, R2, t2):
    """Compute a simple pose distance: rotation + translation norm"""
    rot_diff = torch.norm(R1 - R2)
    trans_diff = torch.norm(t1 - t2)
    return rot_diff + trans_diff

class TwoStagePoseEstimator(nn.Module):
    def __init__(self, sensory_channels, renderer_config, encoder_type="resnet", num_candidates=32, noise_level=0.05, conf_thresh=0.5):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality="rgb", out_dim=64)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels, ngf=64)

        self.det_head = CandidatePoseModel(feature_dim=self.encoder.out_dim)
        self.refiner = CosyPoseStyleRenderMatch(
            renderer_config,
            num_candidates=num_candidates
        )

    def extract_features(self, x_dict):
        return self.encoder(x_dict)

    def detect(self, feat_map, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections = self.det_head.decode_poses(outputs, top_k=top_k)
        return [d for d in detections if d["score"] >= self.conf_thresh], outputs
    
    def sample_candidates(self, detections):
        candidates_all = []
        for det in detections:
            # Base pose (no noise)
            base_q = det["quat"].unsqueeze(0)
            base_t = det["trans"].unsqueeze(0)

            # Generate noisy versions
            noisy_q = F.normalize(
                base_q + torch.randn((self.num_candidates - 1, 4), device=base_q.device) * self.noise_level,
                dim=-1
            )
            noisy_t = base_t + torch.randn((self.num_candidates - 1, 3), device=base_t.device) * self.noise_level

            all_q = torch.cat([base_q, noisy_q], dim=0)  # (N, 4)
            all_t = torch.cat([base_t, noisy_t], dim=0)  # (N, 3)

            R = quaternion_to_matrix(all_q)
            poses = torch.eye(4, device=base_q.device).repeat(self.num_candidates, 1, 1)
            poses[:, :3, :3] = R
            poses[:, :3, 3] = all_t

            candidates_all.append(poses)
        return candidates_all


    def score_candidates(self, feat_map, K, candidates_all, cad_model_lookup):
        all_results = []
        for obj_id, (cad_data, poses) in cad_model_lookup.items():
            scores = self.refiner(
                feat_map=feat_map,
                cad_model_data_list=[cad_data],
                candidate_pose_list=[poses],
                instance_id_list=[obj_id],
                K=K
            )
            all_results.extend(scores)
        return all_results

    def forward(self, x_dict, K, cad_model_lookup, top_k=20):
        feat_map = self.extract_features(x_dict)
        detections, _ = self.detect(feat_map, top_k=top_k)
        candidates_per_det = self.sample_candidates(detections)

        return self.score_candidates(
            feat_map=feat_map,
            K=K,
            candidates_all=candidates_per_det,
            cad_model_lookup={d["obj_id"]: cad_model_lookup[d["obj_id"]] for d in detections}
        )

    def compute_losses(self, x_dict, K, cad_model_lookup, R_gt_list, t_gt_list, instance_id_list):
        """
        Args:
            x_dict: Dictionary with stacked input for the batch.
            K: List of camera intrinsic matrices for each sample in the batch.
            cad_model_lookup: Lookup table for CAD models.
            R_gt_list: Batch of lists, each containing ground truth rotation matrices for each sample.
            t_gt_list: Batch of lists, each containing ground truth translation vectors for each sample.
            instance_id_list: Batch of lists, each containing instance IDs for each sample.
        """
        
        # Step 1: Feature extraction
        feat_map = self.extract_features(x_dict)
        
        # Step 2: Detection and candidate generation
        detections, _ = self.detect(feat_map, top_k=20)
        detections = [d for d in detections if d["score"] >= self.conf_thresh]
        
        # Step 3: Match predictions to GT for each sample in the batch
        matched_detections, matched_gt_R, matched_gt_t, matched_ids = [], [], [], []
        used_indices = set()

        # Iterate over each sample in the batch
        for R_gt_batch, t_gt_batch, instance_id_batch in zip(R_gt_list, t_gt_list, instance_id_list):
            # Ensure that each sample can handle multiple detections
            for R_gt, t_gt, instance_id in zip(R_gt_batch, t_gt_batch, instance_id_batch):
                best_match, best_dist = None, float("inf")
                for j, det in enumerate(detections):
                    if j in used_indices:
                        continue
                    # Compute the distance between the detection and the ground truth
                    dist = pose_distance(
                        quaternion_to_matrix(det["quat"].unsqueeze(0))[0], det["trans"], R_gt, t_gt
                    )
                    if dist < best_dist:
                        best_match = (j, det)
                        best_dist = dist
                if best_match:
                    j, matched = best_match
                    used_indices.add(j)
                    matched_detections.append(matched)
                    matched_gt_R.append(R_gt)
                    matched_gt_t.append(t_gt)
                    matched_ids.append(instance_id)

        if not matched_detections:
            print("[WARN] No valid matches between detections and GT.")
            device = x_dict[next(iter(x_dict))].device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero

        # Step 4: Sample candidates for matching detections
        candidates_all = self.sample_candidates(matched_detections)
        cad_model_data_list = [cad_model_lookup[0] for _ in matched_detections] 
        K_list = [K for _ in matched_detections]  
        feat_maps = [feat_map for _ in matched_detections]  

        # Step 5: Compute loss using refiner
        return self.refiner.compute_losses(
            feat_maps=feat_maps,
            cad_model_data_list=cad_model_data_list,
            candidate_pose_list=candidates_all,
            R_gt_list=matched_gt_R,
            t_gt_list=matched_gt_t,
            instance_id_list=matched_ids,
            K_list=K_list
        )

    def freeze_candidates(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.det_head.parameters():
            param.requires_grad = False

    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False
