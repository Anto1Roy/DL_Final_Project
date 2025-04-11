import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.helpers import quaternion_to_matrix
from Models.ObjectDetection.DetectionHead import CosyPoseDetectionHead

class CandidatePoseModel(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.det_head = CosyPoseDetectionHead(feature_dim)

    def forward(self, feat_map, K=None, extrinsics=None):
        """
        Forward pass.
        Optionally, camera intrinsics (K) and extrinsics (extrinsics) are provided.
        These are passed to the detection head so that it can output predictions in both the original 
        (camera) coordinate system and the transformed (global) coordinates.
        """
        quat, trans, embed, confidence, bbox, global_R, global_trans = self.det_head(
            feat_map, K=K, extrinsics=extrinsics
        )
        return {
            "quat": quat,           # (B, 4, H, W): predicted quaternions (camera coords, normalized)
            "trans": trans,         # (B, 3, H, W): predicted translations in camera coordinates
            "embed": embed,         # (B, D, H, W): per-pixel embedding feature map
            "conf": confidence,     # (B, 1, H, W): predicted confidence scores
            "bbox": bbox,           # (B, 4, H, W): predicted bounding-box regression output
            "global_R": global_R,   # (B, 3, 3): rotation matrices in global coordinates (if extrinsics provided)
            "global_trans": global_trans  # (B, 3): translations in global coordinates (if extrinsics provided)
        }

    def decode_poses(self, outputs, top_k=50):
        """
        Decodes poses from the network outputs. For each sample in the batch, the confidence map is flattened,
        and the top_k predictions are selected. For each selected location, the corresponding quaternion and translation 
        (and bounding box if available) are extracted and used to form a 4x4 pose matrix.
        """
        quat = outputs['quat']
        trans = outputs['trans']
        conf = outputs['conf']
        bbox = outputs.get('bbox', None)

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
                result = {
                    "quat": q,
                    "trans": t,
                    "score": topk_vals[i].item(),
                    "pose_matrix": pose_mat,
                }
                if bbox is not None:
                    result["bbox"] = bbox[b, :, h, w]
                sample_results.append(result)
            results.append(sample_results)
        return results

    def match_and_decode(self, outputs, class_embeddings, top_k=10):
        """
        Optionally uses the embedding branch to match the predictions with given class embeddings.
        """
        quat = outputs['quat']   # [B, 4, H, W]
        trans = outputs['trans'] # [B, 3, H, W]
        embed = outputs['embed'] # [B, D, H, W]
        conf = outputs['conf']   # [B, H, W]

        B, D, H, W = embed.shape
        results = []
        class_names = list(class_embeddings.keys())
        class_matrix = torch.stack([class_embeddings[name] for name in class_names], dim=0).to(embed.device)  # [C, D]
        for b in range(B):
            feat = embed[b].reshape(D, -1)  # [D, H*W]
            sim = torch.matmul(class_matrix, feat)  # [C, H*W]
            topk_sim, topk_idx = sim.view(-1).topk(top_k)
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
