import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ObjectDetection.GridObjectDetector import GridObjectDetector
from Models.Encoding.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.Encoding.ResNet import ResNetFeatureEncoder
from Models.PoseEstimatorSeen.TransformerFusion import TransformerFusion
from Models.helpers import quaternion_to_matrix
from scipy.optimize import linear_sum_assignment


class PoseEstimator(nn.Module):
    def __init__(
        self,
        sensory_channels,
        encoder_type="resnet",
        n_views=1,
        fusion_type=None,
        num_classes=10,
        anchors=3,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.out_dim = 128
        self.ngf = 16

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.ngf, out_dim=self.out_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if fusion_type == "transformer":
            self.fusion_module = TransformerFusion(in_channels=self.out_dim, hidden_dim=64, num_heads=4, num_layers=2)
            self.det_head = GridObjectDetector(in_dim=self.out_dim, num_classes=num_classes, anchors=anchors)
        else:
            self.fusion_module = None
            self.det_head = GridObjectDetector(in_dim=(self.out_dim * n_views), num_classes=num_classes, anchors=anchors)

    def forward(self, x_dict_views, K_list, top_k=100):
        feats = [self.encoder(x) for x in x_dict_views]

        if self.fusion_module:
            fused_feat = self.fusion_module(feats, K_list)
        else:
            fused_feat = torch.cat(feats, dim=1)

        outputs = self.det_head(fused_feat)
        detections = self.det_head.decode_poses(outputs, top_k=top_k)
        return detections, outputs

    def project(self, R_, t_, model_points, K):
        X_cam = R_ @ model_points.T + t_.view(3, 1)
        x_proj = K @ X_cam
        return (x_proj[:2] / x_proj[2]).T

    def compute_reprojection_loss(self, R, t, R_gt, t_gt, K, model_points):
        x_proj_pred = self.project(R, t, model_points, K)
        x_proj_gt = self.project(R_gt, t_gt, model_points, K)
        return F.mse_loss(x_proj_pred, x_proj_gt)

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, gt_obj_ids, K_list, model_points_by_id=None, top_k=100):
        detections, outputs = self.forward(x_dict_views, K_list=K_list, top_k=top_k)
        quat, trans, conf, class_scores = outputs['quat'], outputs['trans'], outputs['conf'], outputs['class_scores']
        B, H, W, A, _ = quat.shape
        device = quat.device
        num_classes = class_scores.shape[-1]

        total_rot_loss, total_trans_loss, total_conf_loss, total_class_loss = 0, 0, 0, 0
        matched_gt = 0

        # Flatten predictions
        pred_quats = quat.view(B, -1, 4)           # (B, N, 4)
        pred_trans = trans.view(B, -1, 3)          # (B, N, 3)
        pred_conf  = conf.view(B, -1)              # (B, N)
        class_scores = class_scores.view(B, -1, num_classes)  # (B, N, C)
        pred_classes = class_scores.argmax(dim=-1)             # (B, N)

        # Only batch size 1 for now
        assert B == 1
        pred_quats = pred_quats[0]
        pred_trans = pred_trans[0]
        pred_conf  = pred_conf[0]
        pred_classes = pred_classes[0]
        class_scores = class_scores[0]

        # Organize GT
        sample_gt = [(int(obj_id.item()), R_gt_list[i], t_gt_list[i]) for i, obj_id in enumerate(gt_obj_ids[0])]
        gt_by_obj = {}
        for obj_id, R, t in sample_gt:
            gt_by_obj.setdefault(obj_id, []).append((R.to(device), t.to(device)))

        for obj_id, gt_instances in gt_by_obj.items():
            gt_Rs = torch.stack([R for R, _ in gt_instances])
            gt_ts = torch.stack([t for _, t in gt_instances])

            # Filter predictions for this class
            mask = (pred_classes == obj_id)
            if not mask.any():
                total_conf_loss += len(gt_Rs)  # all GT missed
                continue

            pred_q_obj = pred_quats[mask]
            pred_t_obj = pred_trans[mask]
            pred_c_obj = pred_conf[mask]
            class_logit_obj = class_scores[mask]

            # Convert quats to matrices
            pred_Rs = quaternion_to_matrix(pred_q_obj)

            # Match each GT to closest pred
            rot_diff = torch.cdist(gt_Rs.reshape(len(gt_Rs), -1), pred_Rs.reshape(len(pred_Rs), -1))
            trans_diff = torch.cdist(gt_ts, pred_t_obj)
            total_diff = rot_diff + trans_diff
            min_costs, indices = total_diff.min(dim=1)  # match GT→pred

            matched_gt += len(gt_Rs)

            for i, j in enumerate(indices):
                R_gt, t_gt = gt_Rs[i], gt_ts[i]
                R_pred, t_pred = pred_Rs[j], pred_t_obj[j]
                q_pred = pred_q_obj[j]

                R_pred = quaternion_to_matrix(pred_quats[j].unsqueeze(0))[0]
                total_rot_loss += F.mse_loss(R_pred, R_gt)
                total_trans_loss += F.mse_loss(t_pred, t_gt)

                total_conf_loss += F.binary_cross_entropy(pred_c_obj[j].unsqueeze(0), torch.tensor([1.0], device=device))
                total_class_loss += F.cross_entropy(class_logit_obj[j].unsqueeze(0), torch.tensor([obj_id], device=device))

        # Penalize unmatched predictions as false positives
        unmatched = torch.ones(pred_conf.shape[0], device=device, dtype=torch.bool)
        for obj_id in gt_by_obj:
            unmatched &= (pred_classes != obj_id)
        total_conf_loss += F.binary_cross_entropy(pred_conf[unmatched], torch.zeros_like(pred_conf[unmatched]))

        if matched_gt == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero, zero

        avg_rot = total_rot_loss / matched_gt
        avg_trans = total_trans_loss / matched_gt
        avg_class = total_class_loss / matched_gt
        avg_conf = total_conf_loss / (matched_gt + unmatched.sum())

        total_loss = avg_rot + avg_trans + avg_class + avg_conf
        return total_loss, avg_rot, avg_trans, avg_class, avg_conf
