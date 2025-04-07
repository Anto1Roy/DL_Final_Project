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

    # ----------- Stage 1: Single-View 6D Pose Estimation -----------
    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)  # (B, C, H, W)

    def detect_single_view(self, feat_map, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections_per_sample = self.det_head.decode_poses(outputs, top_k=top_k)
        filtered = [
            [d for d in dets if d["score"] >= self.conf_thresh]
            for dets in detections_per_sample
        ]
        return filtered, outputs
    
     # ----------- Stage 2: Multi-View Pose Matching -----------
    def match_detections_across_views(self, all_view_detections):
        # NOT IMPLEMENTED: You need to define how to associate objects across views
        # Could involve: object ID matching, appearance feature similarity, geometric constraints
        return NotImplemented
    
    # ----------- Stage 3: Global Scene Refinement -----------
    def global_scene_refinement(self):
        # NOT IMPLEMENTED: Would involve bundle adjustment-like optimization
        return NotImplemented

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

    def compute_losses(self, x_dict, K, cad_model_lookup, R_gt_tensor, t_gt_tensor, instance_id_list, sample_indices):
        device = x_dict[next(iter(x_dict))].device
        B = K.shape[0]  # batch size

        # Extract features for the entire batch
        feat_map = self.extract_features(x_dict)

        # Detect poses for the entire batch
        detections_per_sample, _ = self.detect(feat_map, top_k=25)  # List[List[dict]]

        matched_detections = []
        matched_gt_R, matched_gt_t, matched_ids, matched_feat, matched_K = [], [], [], [], []

        for i in range(len(sample_indices)):
            sample_idx = sample_indices[i].item()

            dets = detections_per_sample[sample_idx]

            best_match, best_dist = None, float('inf')
            for det in dets:
                quat = det["quat"]
                t_pred = det["trans"]
                R_pred = quaternion_to_matrix(quat.unsqueeze(0))[0]
                R_gt = R_gt_tensor[i]
                t_gt = t_gt_tensor[i]

                dist = torch.norm(R_gt - R_pred) + torch.norm(t_gt - t_pred)
                if dist < best_dist:
                    best_dist = dist
                    best_match = det

            if best_match is not None:
                matched_detections.append(best_match)
                matched_gt_R.append(R_gt_tensor[i])
                matched_gt_t.append(t_gt_tensor[i])
                matched_ids.append(instance_id_list[i])
                matched_feat.append(feat_map[sample_idx])
                matched_K.append(K[sample_idx])

        if not matched_detections:
            print("[WARN] No valid matches between detections and GT.")
            return torch.tensor(0.0, device=device), 0.0, 0.0, 0.0

        # Sample candidate poses for each detection
        candidates_all = self.sample_candidates(matched_detections)

        # Retrieve CAD models for each matched detection
        cad_model_data_list = [cad_model_lookup[obj_id] for obj_id in matched_ids]

        # Compute losses with refined poses
        return self.refiner.compute_losses(
            feat_maps=matched_feat,
            cad_model_data_list=cad_model_data_list,
            candidate_pose_list=candidates_all,
            R_gt_list=matched_gt_R,
            t_gt_list=matched_gt_t,
            instance_id_list=matched_ids,
            K_list=matched_K
        )
    
    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.det_head.parameters():
                param.requires_grad = False

