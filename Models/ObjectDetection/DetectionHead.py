import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.helpers import quaternion_to_matrix  # Ensure this function accepts (batch,4) inputs

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

    def forward(self, feat_map, K, extrinsics):
        embed = F.normalize(self.embed_head(feat_map), dim=1)
        pose_out = self.pose_head(feat_map)
        # Channels 0:4 for quaternion
        quat = F.normalize(pose_out[:, 0:4, :, :], dim=1)
        # Channels 4:7 for translation prediction in camera coordinates
        trans = pose_out[:, 4:7, :, :]
        confidence = self.confidence_head(feat_map)

        B, _, H, W = quat.shape

        # Permute to (B, H, W, 4) so that the quaternion channels are the last dimension.
        quat_perm = quat.permute(0, 2, 3, 1)  # shape: (B, H, W, 4)
        # Flatten the batch and spatial dimensions: (B*H*W, 4)
        quat_flat = quat_perm.reshape(-1, 4)
        # Convert flattened quaternions to rotation matrices (expected output: (B*H*W, 3, 3))
        R_flat = quaternion_to_matrix(quat_flat)
        # Reshape back to the original spatial dimensions and permute as desired:
        R_pred = R_flat.reshape(B, H, W, 3, 3).permute(0, 3, 4, 1, 2)  # shape: (B, 3, 3, H, W)

        # -- Reshape translation to (B, 3, H*W) --
        trans_flat = trans.view(B, 3, -1)  # (B, 3, H*W)

        # -- Prepare extrinsics --
        R_w2c = extrinsics["R_w2c"]
        t_w2c = extrinsics["t_w2c"]

        R_w2c = R_w2c.unsqueeze(0)  # now shape becomes (1, 3, 3)
        t_w2c = t_w2c.unsqueeze(0)  # now shape becomes (1, 3)

        # Make sure t_w2c has shape (B, 3, 1) for matrix multiplication.
        t_w2c = t_w2c.unsqueeze(-1)  # (B, 3, 1)

        # Compute the transpose of R_w2c for conversion.
        R_w2c_T = R_w2c.transpose(1, 2)  # (B, 3, 3)

        # --- Compute global rotation conversion ---
        # Assume R_pred has shape (B, 3, 3, H, W).
        B, _, _, H, W = R_pred.shape

        # Reshape R_pred to (B, H*W, 3, 3)
        R_pred_flat = R_pred.reshape(B, 3, 3, -1).permute(0, 3, 1, 2)  # shape: (B, H*W, 3, 3)

        # Multiply: for each pixel, compute R_global = R_w2c^T * R_pred.
        # Here, we unsqueeze R_w2c_T to (B, 1, 3, 3) so that multiplication occurs
        # over the H*W dimension.
        R_global_flat = torch.matmul(R_w2c_T.unsqueeze(1), R_pred_flat)  # shape: (B, H*W, 3, 3)

        # Reshape back to (B, 3, 3, H, W)
        R_global = R_global_flat.permute(0, 2, 3, 1).reshape(B, 3, 3, H, W)

        # --- Compute global translation conversion ---
        # Assume trans_flat has shape (B, 3, H*W) (obtained earlier from 'trans')
        t_global_flat = torch.matmul(R_w2c_T, (trans_flat - t_w2c))  # shape: (B, 3, H*W)
        t_global = t_global_flat.reshape(B, 3, H, W)

        return quat, trans, embed, confidence, R_global, t_global
