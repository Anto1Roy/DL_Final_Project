import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder
from Models.ObjectDetection.GridObjectDetector import GridObjectDetector
from Models.CandidatePose import CandidatePoseModel
from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.helpers import quaternion_to_matrix

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

    def forward(self, x):
        features = self.feature_encoder(x)
        detections = self.object_detector(features)
        return detections

    def compute_losses(self, x_dict_batch, K_batch, cad_model_lookup, R_gt_batch, t_gt_batch, instance_ids_batch):
        feat_maps = self.encoder(x_dict_batch)

        all_scene_feats = []
        all_cad_data = []
        all_candidate_poses = []
        all_instance_ids = []
        all_R_gt = []
        all_t_gt = []

        for i in range(len(feat_maps)):
            feat_map = feat_maps[i]
            K = K_batch[i]
            R_gt_list = R_gt_batch[i]
            t_gt_list = t_gt_batch[i]
            inst_ids = instance_ids_batch[i]
            cad_models = []
            poses = []
            R_gt = []
            t_gt = []
            instance_ids = []

            for j, obj_id in enumerate(inst_ids):
                if obj_id not in cad_model_lookup:
                    continue
                verts, faces = cad_model_lookup[obj_id]
                quat = torch.randn(self.num_candidates, 4, device=feat_map.device)
                quat = F.normalize(quat, dim=-1)
                trans = torch.randn(self.num_candidates, 3, device=feat_map.device) * 0.05
                R = quaternion_to_matrix(quat)
                pose = torch.eye(4, device=feat_map.device).repeat(self.num_candidates, 1, 1)
                pose[:, :3, :3] = R
                pose[:, :3, 3] = trans
                cad_models.append((verts, faces))
                poses.append(pose)
                R_gt.append(R_gt_list[j])
                t_gt.append(t_gt_list[j])
                instance_ids.append(obj_id)

            if poses:
                all_scene_feats.append(feat_map)
                all_cad_data.append(cad_models)
                all_candidate_poses.append(poses)
                all_instance_ids.append(instance_ids)
                all_R_gt.append(R_gt)
                all_t_gt.append(t_gt)

        if len(all_scene_feats) == 0:
            zero = torch.tensor(0.0, requires_grad=True, device=feat_maps[0].device)
            return zero, zero, zero, zero

        return self.refiner.compute_losses(
            all_scene_feats,
            all_cad_data,
            all_candidate_poses,
            all_R_gt,
            all_t_gt,
            all_instance_ids,
            K=K_batch
        )

    def freeze_candidates(self):
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        for param in self.object_detector.parameters():
            param.requires_grad = False
        for param in self.pose_generator.parameters():
            param.requires_grad = False

    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False


        
