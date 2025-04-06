import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.CandidatePose import CandidatePoseModel
from Models.helpers import quaternion_to_matrix, rotation_to_quat

class TwoStagePoseEstimator(nn.Module):
    def __init__(self, sensory_channels, renderer_config, num_candidates=32, noise_level=0.05):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.base_model = CandidatePoseModel(sensory_channels)
        self.refiner = CosyPoseStyleRenderMatch(sensory_channels, renderer_config, num_candidates=num_candidates)

    def detect(self, x_dict, top_k=10):
        return self.base_model.decode_poses(self.base_model(x_dict), top_k=top_k)

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
            candidates_all.append((det["obj_id"], poses))
        return candidates_all

    def score_candidates(self, x_dict, K, candidates_all, cad_model_lookup):
        pose_inputs = []
        obj_ids = []
        for obj_id, poses in candidates_all:
            pose_inputs.append(poses.unsqueeze(0))
            obj_ids.append(obj_id)

        cad_model_data_list = [cad_model_lookup[obj_id] for obj_id in obj_ids]
        instance_id_list = [obj_ids]  # batch size = 1
        return self.refiner(x_dict, cad_model_data_list, pose_inputs, instance_id_list)

    def forward(self, x_dict, K, cad_model_lookup, top_k=10):
        detections = self.detect(x_dict, top_k=top_k)
        candidates_all = self.sample_candidates(detections)
        results = self.score_candidates(x_dict, K, candidates_all, cad_model_lookup)
        return results

    def compute_losses(self, x_dict, K, cad_model_lookup,
                       R_gt_list, t_gt_list, instance_id_list):

        yolo_output = self.base_model(x_dict)
        detections = self.base_model.decode_poses(yolo_output, top_k=len(instance_id_list))

        candidates_all = []
        for det in detections:
            obj_id = det["obj_id"]
            if obj_id not in instance_id_list:
                continue
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

            candidates_all.append((obj_id, poses))

        cad_model_data_list = [cad_model_lookup[obj_id] for obj_id, _ in candidates_all]
        candidate_pose_list = [poses for _, poses in candidates_all]
        matched_ids = [obj_id for obj_id, _ in candidates_all]

        gt_R_list = []
        gt_t_list = []
        for obj_id in matched_ids:
            idx = instance_id_list.index(obj_id)
            gt_R_list.append(R_gt_list[idx])
            gt_t_list.append(t_gt_list[idx])

        return self.refiner.compute_losses(
            x_dict,
            cad_model_data_list=cad_model_data_list,
            candidate_pose_list=candidate_pose_list,
            R_gt_list=gt_R_list,
            t_gt_list=gt_t_list,
            instance_id_list=matched_ids,
            K=K
        )