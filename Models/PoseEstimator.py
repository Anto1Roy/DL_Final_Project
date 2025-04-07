import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.CandidatePose import CandidatePoseModel
from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder  # or FuseNetFeatureEncoder if preferred
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

        self.out_dim = 64

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(modality=list(sensory_channels.keys()), out_dim=self.out_dim)

        self.det_head = CandidatePoseModel(feature_dim=self.out_dim)
        self.refiner = CosyPoseStyleRenderMatch(
            renderer_config,
            num_candidates=num_candidates,
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

    def forward(self, x_dict, K, cad_model_lookup, top_k=100):
        feat_map = self.extract_features(x_dict)
        detections, _ = self.detect(feat_map, top_k=top_k)
        candidates_per_det = self.sample_candidates(detections)

        candidates_all = {id(poses): poses for poses in candidates_per_det}
        results = self.score_candidates(feat_map, K, candidates_all, cad_model_lookup)
        return results

    def compute_losses(self, x_dict, K, cad_model_lookup, R_gt_list, t_gt_list, instance_id_list):
        # Extract features for the entire batch
        feat_map = self.extract_features(x_dict)
        
        # Detect the top_k detections for each sample in the batch
        detections, _ = self.detect(feat_map, top_k=25)  # List of lists of detections, one per sample
        
        matched_detections, matched_gt_R, matched_gt_t, matched_ids = [], [], [], []
        used_indices = set()

        # Iterate over the batch (one sample at a time)
        for i in range(len(R_gt_list)):
            R_gt = R_gt_list[i]
            t_gt = t_gt_list[i]
            obj_id = instance_id_list[i]

            # Get the detections for the current sample (i)
            sample_detections = detections[i]  # List of detections for sample i
            
            best_match, best_dist = None, float("inf")
            
            # Compare GT with the detections of the current sample
            for j, det in enumerate(sample_detections):
                if j in used_indices:
                    continue
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
                matched_ids.append(obj_id)

        # If no valid detections were matched, return 0 loss
        if not matched_detections:
            print("[WARN] No valid matches between detections and GT.")
            return torch.tensor(0.0, device=x_dict[next(iter(x_dict))].device), 0.0, 0.0, 0.0

        # Sample candidate poses for the matched detections
        candidates_all = self.sample_candidates(matched_detections)
        
        # Create the cad_model_data_list for each matched detection in the batch
        cad_model_data_list = [cad_model_lookup[obj_id] for obj_id in matched_ids]
        
        # Call the refiner's compute_losses for batch processing
        return self.refiner.compute_losses(
            feat_map=feat_map,
            cad_model_data_list=cad_model_data_list,
            candidate_pose_list=candidates_all,
            R_gt_list=matched_gt_R,
            t_gt_list=matched_gt_t,
            instance_id_list=matched_ids,
            K=K
        )
