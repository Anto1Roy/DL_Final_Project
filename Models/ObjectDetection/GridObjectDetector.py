import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix

class GridObjectDetector(nn.Module):
    def __init__(self, in_dim, num_classes=10, anchors=3):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        out_dim = anchors * (7 + 1 + num_classes)  # quat(4) + trans(3) + conf + cls
        self.head = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, feat_map):
        # B x (A * (8+num_classes)) x H x W
        out = self.head(feat_map)
        B, _, H, W = out.shape
        out = out.view(B, self.anchors, 8 + self.num_classes, H, W).permute(0, 3, 4, 1, 2).contiguous()
        return out  # (B, H, W, A, 8+num_classes)

    def apply_activations(self, out):
        return {
            "quat": F.normalize(out[..., 0:4], dim=-1),
            "trans": out[..., 4:7],
            "conf": out[..., 7], 
            "class_scores": F.softmax(out[..., 8:], dim=-1),
            "raw": out
        }

    def decode_poses(self, outputs, top_k=10):
        B, H, W, A, _ = outputs["quat"].shape
        results = []
        for b in range(B):
            flat_conf = outputs["conf"][b].reshape(-1)
            topk_vals, topk_idx = flat_conf.topk(top_k)
            for i in range(top_k):
                idx = topk_idx[i].item()
                h = idx // (W * A)
                w = (idx % (W * A)) // A
                a = idx % A
                q = outputs["quat"][b, h, w, a]
                t = outputs["trans"][b, h, w, a]
                cls_probs = outputs["class_scores"][b, h, w, a]
                obj_id = torch.argmax(cls_probs).item()
                pose_mat = torch.eye(4, device=q.device)
                pose_mat[:3, :3] = quaternion_to_matrix(q.unsqueeze(0))[0]
                pose_mat[:3, 3] = t
                results.append({
                    "batch_idx": b, "obj_id": obj_id, "quat": q,
                    "trans": t, "score": topk_vals[i].item(), "pose_matrix": pose_mat
                })
        return results

