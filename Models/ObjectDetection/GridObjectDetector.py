import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.helpers import quaternion_to_matrix

class GridObjectDetector(nn.Module):
    """
    Anchor-based object detector that predicts:
      - Quaternions (4D) for pose orientation
      - Translations (3D) for pose position
      - Confidence (1D) for object presence
      - Class probabilities (num_classes)

    By default, each output cell has 'anchors' anchor boxes.
    The model reshapes outputs into a (B, H, W, anchors, 8+num_classes)
    layout, where:
      - 4D quaternion
      - 3D translation
      - 1D confidence
      - num_classes class probabilities
    """
    def __init__(self, in_dim, num_classes=10, anchors=3):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.in_dim = in_dim

        self.head = nn.Conv2d(
            in_channels=self.in_dim,
            out_channels=anchors * (7 + 1 + num_classes), 
            kernel_size=1
        )

    def forward(self, feat_map):
        
        out = self.head(feat_map)               # (B, A*(7+1+num_classes), H', W')
        B, _, H, W = out.shape
        
        out = out.view(B, self.anchors, 8 + self.num_classes, H, W)
        out = out.permute(0, 3, 4, 1, 2).contiguous()
        # Now out shape = (B, H, W, anchors, 8+num_classes)

        quat = F.normalize(out[..., 0:4], dim=-1)      # shape (B, H, W, A, 4)
        trans = out[..., 4:7]                          # shape (B, H, W, A, 3)
        conf = torch.sigmoid(out[..., 7])              # shape (B, H, W, A)
        
        class_scores = F.softmax(out[..., 8:], dim=-1) # shape (B, H, W, A, num_classes)

        return {
            "quat": quat,             # (B, H, W, A, 4)
            "trans": trans,           # (B, H, W, A, 3)
            "conf": conf,             # (B, H, W, A)
            "class_scores": class_scores,  # (B, H, W, A, num_classes)
            "raw": out                # (B, H, W, A, 8+num_classes) if needed
        }

    def decode_poses(self, outputs, top_k=10):
        """
        Returns the top-K detections across all anchors and grid cells.
        Each detection includes:
          - obj_id
          - quaternion (quat)
          - translation (trans)
          - confidence score
          - 4x4 pose matrix

        If B > 1, this merges them into a single list.
        You can adjust it to return per-sample lists if desired.
        """
        quat = outputs['quat']         # (B, H, W, A, 4)
        trans = outputs['trans']       # (B, H, W, A, 3)
        conf = outputs['conf']         # (B, H, W, A)
        class_scores = outputs['class_scores']  # (B, H, W, A, num_classes)

        B, H, W, A, _ = quat.shape
        results = []

        # Flatten over H, W, A to pick top K across the entire batch
        for b in range(B):
            flat_conf = conf[b].reshape(-1)             # shape (H*W*A)
            topk_vals, topk_idx = flat_conf.topk(top_k)

            for i in range(top_k):
                idx = topk_idx[i].item()

                # Reconstruct (h, w, a)
                h = idx // (W * A)
                w = (idx % (W * A)) // A
                a = idx % A

                # Extract quaternion, translation, class
                q = quat[b, h, w, a]              # shape (4,)
                t = trans[b, h, w, a]             # shape (3,)
                cls_probs = class_scores[b, h, w, a]
                obj_id = torch.argmax(cls_probs).item()

                # Build 4x4 pose matrix
                pose_mat = torch.eye(4, device=q.device)
                pose_mat[:3, :3] = quaternion_to_matrix(q.unsqueeze(0))[0]
                pose_mat[:3, 3] = t

                results.append({
                    "batch_idx": b,
                    "obj_id": obj_id,
                    "quat": q,
                    "trans": t,
                    "score": topk_vals[i].item(),
                    "pose_matrix": pose_mat
                })

        return results
