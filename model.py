import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.encoder(x).squeeze(-1).squeeze(-1)  # (B, C)


class PoseEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # multimodal branches (shared weights across views)
        self.rgb_branch = CNNBranch(3)
        self.depth_branch = CNNBranch(1)
        self.aolp_branch = CNNBranch(1)
        self.dolp_branch = CNNBranch(1)

        self.fusion_proj = nn.Linear(128 * 4, 256)  # concat modal features
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.pose_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 7)  # quaternion (4) + translation (3)
        )

    def forward(self, imgs, depths, aolps, dolps, K=None, R_gt=None, t_gt=None):
        B, V, C, H, W = imgs.shape  # batch, views, channels, height, width
        features = []
        for v in range(V):
            rgb_feat = self.rgb_branch(imgs[:, v])
            d_feat = self.depth_branch(depths[:, v])
            a_feat = self.aolp_branch(aolps[:, v])
            do_feat = self.dolp_branch(dolps[:, v])
            modal_feat = torch.cat([rgb_feat, d_feat, a_feat, do_feat], dim=1)  # (B, 512)
            features.append(self.fusion_proj(modal_feat))

        view_features = torch.stack(features, dim=0)  # (V, B, 256)
        fused = self.view_transformer(view_features)  # (V, B, 256)
        fused = fused.mean(dim=0)  # (B, 256)

        pose_vec = self.pose_head(fused)  # (B, 7)
        quat = F.normalize(pose_vec[:, :4], dim=1)  # Normalize quaternion
        trans = pose_vec[:, 4:]

        if R_gt is not None and t_gt is not None:
            rot_loss, trans_loss = self.compute_loss(quat, trans, R_gt, t_gt)
            return quat, trans, rot_loss, trans_loss
        return quat, trans

    def compute_loss(self, pred_q, pred_t, gt_R, gt_t):
        # Convert pred_q to rotation matrix
        B = pred_q.shape[0]
        pred_R = self.quaternion_to_matrix(pred_q)
        rot_loss = F.mse_loss(pred_R, gt_R)
        trans_loss = F.mse_loss(pred_t, gt_t)
        return rot_loss, trans_loss

    def quaternion_to_matrix(self, q):
        # q: (B, 4) quaternion -> (B, 3, 3)
        B = q.shape[0]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros((B, 3, 3), device=q.device)
        R[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
        R[:, 0, 1] = 2 * (qx * qy - qz * qw)
        R[:, 0, 2] = 2 * (qx * qz + qy * qw)
        R[:, 1, 0] = 2 * (qx * qy + qz * qw)
        R[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
        R[:, 1, 2] = 2 * (qy * qz - qx * qw)
        R[:, 2, 0] = 2 * (qx * qz - qy * qw)
        R[:, 2, 1] = 2 * (qy * qz + qx * qw)
        R[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
        return R