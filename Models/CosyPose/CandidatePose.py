import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix
from Models.ObjectDetection.DetectionHead import CosyPoseDetectionHead


class CandidatePoseModel(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.det_head = CosyPoseDetectionHead(feature_dim)

    def forward(self, feat_map):
        quat, trans, embed, confidence = self.det_head(feat_map)
        return {
            "quat": quat,             # (B, 4, H, W)
            "trans": trans,           # (B, 3, H, W)
            "embed": embed,           # (B, D, H, W)
            "conf": confidence              # (B, H, W)
        }

    def decode_poses(self, outputs, top_k=10):
        quat = outputs['quat']
        trans = outputs['trans']
        # embed = outputs['embed']   # [B, D, H, W] (not used in this function)
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
    
    def match_and_decode(self, outputs, class_embeddings, top_k=10):
        quat = outputs['quat']     # [B, 4, H, W]
        trans = outputs['trans']   # [B, 3, H, W]
        embed = outputs['embed']   # [B, D, H, W]
        conf = outputs['conf']     # [B, H, W]

        B, D, H, W = embed.shape
        results = []

        # Convert class_embeddings to tensor
        class_names = list(class_embeddings.keys())
        class_matrix = torch.stack([class_embeddings[name] for name in class_names], dim=0).to(embed.device)  # [C, D]

        for b in range(B):
            feat = embed[b].reshape(D, -1)  # [D, H*W]
            sim = torch.matmul(class_matrix, feat)  # [C, H*W]

            topk_sim, topk_idx = sim.view(-1).topk(top_k)  # Flattened

            sample_results = []
            for i in range(top_k):
                flat_idx = topk_idx[i].item()
                cls_idx = flat_idx // (H * W)
                pixel_idx = flat_idx % (H * W)
                h = pixel_idx // W
                w = pixel_idx % W

                q = quat[b, :, h, w]
                t = trans[b, :, h, w]

                pose_mat = torch.eye(4, device=q.device)
                pose_mat[:3, :3] = quaternion_to_matrix(q.unsqueeze(0))[0]
                pose_mat[:3, 3] = t

                sample_results.append({
                    "class": class_names[cls_idx],
                    "quat": q,
                    "trans": t,
                    "score": topk_sim[i].item(),
                    "pose_matrix": pose_mat
                })

            results.append(sample_results)

        return results

