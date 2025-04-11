import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.helpers import quaternion_to_matrix

class CosyPoseDetectionHead(nn.Module):
    def __init__(self, feature_dim, embed_dim=128):
        super().__init__()
        self.embed_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=1)
        )
        self.pose_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 7, kernel_size=1)
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 2D bounding box to improve loss convergence
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1)
        )

    def forward(self, feat_map, K=None, extrinsics=None):
        """
        Forward pass.
        
        Args:
            feat_map: Input feature map (B, C, H, W).
            K (optional): Camera intrinsics (B, 3, 3); currently not used explicitly.
            extrinsics (optional): Dictionary with keys 'R_w2c' (B, 3, 3) and 't_w2c' (B, 3)
                                   representing the camera-to-world extrinsics.
                                   
        Returns:
            quat: Predicted quaternions (B, 4, H, W) in camera coordinates (or global if extrinsics applied).
            trans: Predicted translations (B, 3, H, W) in camera coordinates (or global if extrinsics applied).
            embed: Embedding features (B, D, H, W).
            confidence: Confidence scores (B, 1, H, W).
            bbox: Bounding box predictions (B, 4, H, W).
            global_R: Global rotation matrices (B, 3, 3) if extrinsics provided, else None.
            global_trans: Global translations (B, 3) if extrinsics provided, else None.
        """
        embed = F.normalize(self.embed_head(feat_map), dim=1)
        pose_out = self.pose_head(feat_map)
        quat = F.normalize(pose_out[:, 0:4, :, :], dim=1)
        trans = pose_out[:, 4:7, :, :]
        confidence = self.confidence_head(feat_map)
        bbox = self.bbox_head(feat_map) 

        global_R = None
        global_trans = None

        B = feat_map.shape[0]
        global_R_list = []
        global_trans_list = []
        R_w2c = extrinsics["R_w2c"]  # (B, 3, 3)
        t_w2c = extrinsics["t_w2c"]  # (B, 3)
        for b in range(B):
            H, W = quat.shape[2], quat.shape[3]
            center_h = H // 2
            center_w = W // 2
            q_center = quat[b, :, center_h, center_w]
            R_cam = quaternion_to_matrix(q_center.unsqueeze(0))[0] 
            R_global = torch.matmul(R_w2c[b].T, R_cam)
            t_center = trans[b, :, center_h, center_w]
            t_global = torch.matmul(R_w2c[b].T, (t_center - t_w2c[b]).unsqueeze(-1)).squeeze(-1)
            global_R_list.append(R_global.unsqueeze(0))
            global_trans_list.append(t_global.unsqueeze(0))
        global_R = torch.cat(global_R_list, dim=0)  # (B, 3, 3)
        global_trans = torch.cat(global_trans_list, dim=0)  # (B, 3)

        return quat, trans, embed, confidence, bbox, global_R, global_trans
