import torch
import torch.nn as nn
import torch.nn.functional as F


from Models.helpers import quaternion_to_matrix


class GridObjectDetector(nn.Module):
    def __init__(self, encoder, num_classes=10, anchors=3):
        super().__init__()
        self.encoder = encoder
        self.anchors = anchors
        self.num_classes = num_classes
        self.out_dim = encoder.out_dim
        self.head = nn.Conv2d(self.out_dim, anchors * (7 + 1 + num_classes), kernel_size=1)

    def forward(self, x_dict):
        feat_map = self.encoder(x_dict)  # (B, C, H, W)
        out = self.head(feat_map)  # (B, A*(7+1+C), H, W)
        B, _, H, W = out.shape

        out = out.view(B, self.anchors, 8 + self.num_classes, H, W)
        out = out.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, A, 8+C)

        quat = F.normalize(out[..., 0:4], dim=-1)
        trans = out[..., 4:7]
        conf = torch.sigmoid(out[..., 7])
        class_scores = F.softmax(out[..., 8:], dim=-1)

        return {
            "quat": quat,
            "trans": trans,
            "conf": conf,
            "class_scores": class_scores,
            "raw": out
        }

    def decode_poses(self, outputs, top_k=10):
        quat = outputs['quat']
        trans = outputs['trans']
        conf = outputs['conf']
        class_scores = outputs['class_scores']

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
