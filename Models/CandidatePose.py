import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix
from Models.ObjectDetection.CosyPose_Encoder import CosyPoseDetectionHead


class CandidatePoseModel(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.det_head = CosyPoseDetectionHead(feature_dim)

    def forward(self, feat_map):
        quat, trans, conf = self.det_head(feat_map)
        return {
            "quat": quat,             # (B, 4, H, W)
            "trans": trans,           # (B, 3, H, W)
            "conf": conf              # (B, H, W)
        }

    def decode_poses(self, outputs, top_k=10):
        quat = outputs['quat']
        trans = outputs['trans']
        conf = outputs['conf']

        B, _, H, W = quat.shape
        results = []

        for b in range(B):
            flat_conf = conf[b].reshape(-1)
            topk_vals, topk_idx = flat_conf.topk(top_k)

            sample_results = []
            for i in range(top_k):
                idx = topk_idx[i].item()
                h = idx // W
                w = idx % W

                q = quat[b, :, h, w]
                t = trans[b, :, h, w]

                pose_mat = torch.eye(4, device=q.device)
                pose_mat[:3, :3] = quaternion_to_matrix(q.unsqueeze(0))[0]
                pose_mat[:3, 3] = t

                sample_results.append({
                    "quat": q,
                    "trans": t,
                    "score": topk_vals[i].item(),
                    "pose_matrix": pose_mat
                })

            results.append(sample_results) 

        return results  

