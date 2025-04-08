import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from Models.MultiViewMatcher import MultiViewMatcher
from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.CandidatePose import CandidatePoseModel
from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder
from Models.SceneRefinder import SceneRefiner
from Models.helpers import quaternion_to_matrix

def pose_distance(R1, t1, R2, t2):
    rot_diff = torch.norm(R1 - R2)
    trans_diff = torch.norm(t1 - t2)
    return rot_diff + trans_diff


class TwoStagePoseEstimator(nn.Module):
    def __init__(self, sensory_channels, renderer_config, encoder_type="resnet", num_candidates=32, noise_level=0.05, conf_thresh=0.5):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh
        self.out_dim = 16

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.out_dim)

        self.det_head = CandidatePoseModel(feature_dim=self.out_dim)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect_single_view(self, feat_map, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections_per_sample = self.det_head.decode_poses(outputs, top_k=top_k)
        filtered = [
            [d for d in dets if d["score"] >= self.conf_thresh]
            for dets in detections_per_sample
        ]
        return filtered, outputs
    
    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, K_list):
        device = K_list[0].device

        # Extract features per view
        feat_maps = [
            self.extract_single_view_features(
                {mod: view[mod].unsqueeze(0) for mod in view}
            ) for view in x_dict_views
        ]

        # Stack all GT poses across views (they're the same for all views)
        gt_Rs = torch.stack(R_gt_list).to(device)  # shape: (num_gt, 3, 3)
        gt_ts = torch.stack(t_gt_list).to(device)  # shape: (num_gt, 3)

        all_pred_Rs = []
        all_pred_ts = []

        # Collect all detections from all views
        for i, feat_map in enumerate(feat_maps):
            detections_batch, _ = self.detect_single_view(feat_map, top_k=5)
            detections = detections_batch[0]

            if len(detections) == 0:
                continue

            pred_Rs = torch.stack([
                quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
                for det in detections
            ])
            pred_ts = torch.stack([
                det['trans'] for det in detections
            ])

            all_pred_Rs.append(pred_Rs)
            all_pred_ts.append(pred_ts)

        if len(all_pred_Rs) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        # Combine all detections from all views
        all_pred_Rs = torch.cat(all_pred_Rs, dim=0)  # shape: (N_pred, 3, 3)
        all_pred_ts = torch.cat(all_pred_ts, dim=0)  # shape: (N_pred, 3)

        num_gt = len(gt_Rs)
        num_pred = len(all_pred_Rs)

        # Compute pairwise distances
        rot_diff = torch.cdist(gt_Rs.view(num_gt, -1), all_pred_Rs.view(num_pred, -1))
        trans_diff = torch.cdist(gt_ts.view(num_gt, -1), all_pred_ts.view(num_pred, -1))

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost_matrix = (rot_diff + trans_diff).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        losses_rot = []
        losses_trans = []

        for gt_idx, pred_idx in zip(row_ind, col_ind):
            losses_rot.append(rot_diff[gt_idx, pred_idx])
            losses_trans.append(trans_diff[gt_idx, pred_idx])

        losses_rot = torch.stack(losses_rot)
        losses_trans = torch.stack(losses_trans)
        total_loss = losses_rot.mean() + losses_trans.mean()

        return total_loss, losses_rot.mean(), losses_trans.mean()

        

    def forward(self, x_dict_list, K_list, cad_model_lookup, top_k=100):
        # Stage 1: extract features + detections
        feat_maps = [self.extract_single_view_features(x_dict) for x_dict in x_dict_list]
        detections_per_view = [self.detect_single_view(f, top_k=top_k)[0][0] for f in feat_maps]

    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.det_head.parameters():
            param.requires_grad = False