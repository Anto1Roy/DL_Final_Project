import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.ObjectDetection.GridObjectDetector import GridObjectDetector
from Models.Encoding.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.Encoding.ResNet import ResNetFeatureEncoder
from Models.PoseEstimatorSeen.TransformerFusion import TransformerFusion
from Models.helpers import hungarian_matching, quaternion_to_matrix
from scipy.optimize import linear_sum_assignment


class PoseEstimator(nn.Module):
    def __init__(
        self,
        sensory_channels,
        encoder_type="resnet",
        n_views=1,
        fusion_type=None,
        obj_ids={},
        anchors=3,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.out_dim = 128
        self.ngf = 16
        self.obj_ids = sorted(list(obj_ids))  # Ensure consistent ordering
        self.obj_id_to_class_idx = {obj_id: idx for idx, obj_id in enumerate(self.obj_ids)}
        self.class_idx_to_obj_id = {idx: obj_id for obj_id, idx in self.obj_id_to_class_idx.items()}

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.ngf, out_dim=self.out_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if fusion_type == "transformer":
            self.fusion_module = TransformerFusion(in_channels=self.out_dim, hidden_dim=self.out_dim, num_heads=4, num_layers=2)
            self.det_head = GridObjectDetector(in_dim=self.out_dim, num_classes=len(obj_ids), anchors=anchors)
        else:
            self.fusion_module = None
            self.det_head = GridObjectDetector(in_dim=(self.out_dim * n_views), num_classes=len(obj_ids), anchors=anchors)

    def forward(self, x_dict_views, K_list, top_k=100):
        feats = [self.encoder(x) for x in x_dict_views]

        if self.fusion_module:
            fused_feat = self.fusion_module(feats, K_list)
        else:
            fused_feat = torch.cat(feats, dim=1)

        raw_outputs = self.det_head(fused_feat)
        outputs = self.det_head.apply_activations(raw_outputs)
        detections = self.det_head.decode_poses(outputs, top_k=top_k)
        return detections, raw_outputs

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, gt_obj_ids, K_list, top_k=100):
        activations, raw_out = self.forward(x_dict_views, K_list=K_list, top_k=top_k)
        outputs = self.det_head.apply_activations(raw_out)

        quat = outputs['quat'][0].reshape(-1, 4)
        trans = outputs['trans'][0].reshape(-1, 3)
        conf = outputs['conf'][0].reshape(-1)
        class_scores = outputs['class_scores'][0].reshape(-1, outputs['class_scores'].shape[-1])
        pred_classes = class_scores.argmax(dim=-1)
        pred_Rs = quaternion_to_matrix(quat)

        device = quat.device
        matched_pred_idx = set()
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        total_class_loss = 0.0
        total_conf_loss = 0.0
        total_matches = 0

        # Group GTs by object ID
        gt_by_class = {}
        for i, obj_id in enumerate(gt_obj_ids):
            gt_by_class.setdefault(int(obj_id.item()), []).append((R_gt_list[i].to(device), t_gt_list[i].to(device)))

        for obj_id, gt_poses in gt_by_class.items():
            gt_Rs = torch.stack([R for R, _ in gt_poses])
            gt_ts = torch.stack([t for _, t in gt_poses])

            mask = pred_classes == obj_id
            if mask.sum() == 0:
                continue

            pred_R_obj = pred_Rs[mask]
            pred_t_obj = trans[mask]
            pred_conf_obj = conf[mask]
            pred_class_logits = class_scores[mask]
            pred_indices = mask.nonzero(as_tuple=True)[0]

            rot_diff = torch.cdist(gt_Rs.view(len(gt_Rs), -1), pred_R_obj.view(len(pred_R_obj), -1))
            trans_diff = torch.cdist(gt_ts, pred_t_obj)
            cost_matrix = rot_diff + trans_diff  # [N_gt, N_pred]

            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            for i, j in zip(row_ind, col_ind):
                R_gt, t_gt = gt_poses[i]
                R_pred, t_pred = pred_R_obj[j], pred_t_obj[j]

                total_rot_loss += F.mse_loss(R_pred, R_gt)
                total_trans_loss += F.mse_loss(t_pred, t_gt)
                class_idx = torch.tensor([self.obj_id_to_class_idx[obj_id]], device=device)
                total_class_loss += F.cross_entropy(
                    pred_class_logits[j].unsqueeze(0), class_idx
                )
                target_conf = class_scores[pred_indices[j], obj_id] 
                total_conf_loss += F.binary_cross_entropy_with_logits(
                    pred_conf_obj[j].unsqueeze(0), target_conf.unsqueeze(0)
                )
                matched_pred_idx.add(pred_indices[j].item())
                total_matches += 1

        unmatched_mask = torch.ones(len(conf), dtype=torch.bool, device=device)
        if matched_pred_idx:
            unmatched_mask[list(matched_pred_idx)] = False
        if unmatched_mask.any():
            total_conf_loss += F.binary_cross_entropy_with_logits(
                conf[unmatched_mask], torch.zeros_like(conf[unmatched_mask])
            )

        if total_matches == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero, zero, zero, zero

        avg_rot = total_rot_loss / total_matches
        avg_trans = total_trans_loss / total_matches
        avg_class = total_class_loss / total_matches
        avg_conf = total_conf_loss / len(conf)
        
        total_loss = avg_rot + avg_trans + avg_class + avg_conf

        return total_loss, avg_rot, avg_trans, avg_class, avg_conf

    
    def project(self, R_, t_, model_points, K, eps=1e-6):
        X_cam = R_ @ model_points.T + t_.view(3, 1)     # (3, N)
        x_proj = K @ X_cam                              # (3, N)
        z = x_proj[2].clamp(min=eps)                    # Prevent divide-by-zero
        return (x_proj[:2] / z).T                # (N, 2)

    def compute_reprojection_loss(self, R, t, R_gt, t_gt, K, model_points):
        x_proj_pred = self.project(R, t, model_points, K)
        x_proj_gt = self.project(R_gt, t_gt, model_points, K)
        return F.mse_loss(x_proj_pred, x_proj_gt)





