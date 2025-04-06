import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix
from Models.FuseEncoder import FuseEncoder
from Models.FuseDecoder import FuseDecoder

class CandidatePoseModel(nn.Module):
    def __init__(self, sensory_channels, num_classes=10, anchors=3, ngf=64):
        super().__init__()
        self.encoder = FuseEncoder(sensory_channels, ngf=ngf)
        self.decoder = FuseDecoder(ngf=ngf)

        self.anchors = anchors
        self.num_classes = num_classes
        
        self.head = nn.Conv2d(ngf, anchors * (7 + 1 + num_classes), kernel_size=1)

    def forward(self, x_dict):
        x_latent, skip_feats = self.encoder(x_dict)
        feat_map = self.decoder(x_latent, skip_feats)  # (B, ngf, H, W)

        out = self.head(feat_map)  # (B, A*(7+1+C), H, W)
        B, _, H, W = out.shape

        out = out.view(B, self.anchors, 8 + self.num_classes, H, W)  # (B, A, 8+C, H, W)
        out = out.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, A, 8+C)

        # Split outputs
        quat = F.normalize(out[..., 0:4], dim=-1)  # (B, H, W, A, 4)
        trans = out[..., 4:7]                     # (B, H, W, A, 3)
        conf = torch.sigmoid(out[..., 7])         # (B, H, W, A)
        class_scores = F.softmax(out[..., 8:], dim=-1)  # (B, H, W, A, C)

        return {
            "quat": quat,
            "trans": trans,
            "conf": conf,
            "class_scores": class_scores,
            "raw": out  # for loss computation if needed
        }

    def decode_poses(self, yolo_output, top_k=10):
        """
        Decode top K detections from model output.
        Returns list of dicts with keys: obj_id, quat, trans, score, pose_matrix
        """
        quat = yolo_output['quat']
        trans = yolo_output['trans']
        conf = yolo_output['conf']
        class_scores = yolo_output['class_scores']

        B, H, W, A, _ = quat.shape
        results = []

        for b in range(B):
            flat_conf = conf[b].reshape(-1)
            topk_vals, topk_idx = flat_conf.topk(top_k)

            for i in range(top_k):
                idx = topk_idx[i].item()
                h = idx // (W * A)
                w = (idx % (W * A)) // A
                a = idx % A

                q = quat[b, h, w, a]
                t = trans[b, h, w, a]
                cls_probs = class_scores[b, h, w, a]
                obj_id = torch.argmax(cls_probs).item()

                pose_mat = torch.eye(4, device=q.device)
                pose_mat[:3, :3] = quaternion_to_matrix(q.unsqueeze(0))[0]
                pose_mat[:3, 3] = t

                results.append({
                    "obj_id": obj_id,
                    "quat": q,
                    "trans": t,
                    "score": topk_vals[i].item(),
                    "pose_matrix": pose_mat
                })
        return results
