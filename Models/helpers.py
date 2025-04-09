import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim, out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

def quaternion_to_matrix_no_B(q):
    """
    Converts a batch of quaternions to rotation matrices.
    Args:
        q: (N, 4) quaternion in (x, y, z, w) format
    Returns:
        R: (N, 3, 3) rotation matrices
    """
    q = F.normalize(q, dim=-1)  # Ensure unit quaternion

    qx, qy, qz, qw = q.unbind(-1)

    # Precompute terms
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    ww = qw * qw
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - zw)
    R[:, 0, 2] = 2 * (xz + yw)

    R[:, 1, 0] = 2 * (xy + zw)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - xw)

    R[:, 2, 0] = 2 * (xz - yw)
    R[:, 2, 1] = 2 * (yz + xw)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R

def quaternion_to_matrix(q):
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

def rotation_to_quat(R):
    q = Rotation.from_matrix(R.cpu().numpy()).as_quat()
    return torch.tensor(q, dtype=torch.float32, device=R.device)

class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super().__init__()
        self.conv_1 = conv_block(in_dim, out_dim, act_fn)
        self.conv_2 = conv_block_3(out_dim, out_dim, act_fn)
        self.conv_3 = conv_block(out_dim, out_dim, act_fn)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

def compute_add_s(pred_R, pred_t, gt_R, gt_t, model_points, sym_permutations=None, reduction='mean'):
    """
    Computes ADD-S (Average Distance of model points - Symmetric) between predicted and GT poses.
    
    Args:
        pred_R (B, 3, 3): predicted rotation matrix
        pred_t (B, 3): predicted translation vector
        gt_R (B, 3, 3): ground truth rotation matrix
        gt_t (B, 3): ground truth translation vector
        model_points (B, N, 3): model points in object space
        sym_permutations (optional): list of (B, N, 3) tensors representing symmetrical variants
        reduction: 'mean' or 'none'

    Returns:
        add_s (B,): per-object ADD-S distances
    """
    B, N, _ = model_points.shape

    # Transform model points using predicted and GT poses
    pred_pts = (pred_R @ model_points.transpose(1, 2)).transpose(1, 2) + pred_t[:, None, :]  # (B, N, 3)
    gt_pts = (gt_R @ model_points.transpose(1, 2)).transpose(1, 2) + gt_t[:, None, :]        # (B, N, 3)

    if sym_permutations is None:
        # Euclidean distance per point
        dists = torch.norm(pred_pts - gt_pts, dim=-1)  # (B, N)
        add_s = dists.mean(dim=-1)                     # (B,)
    else:
        min_dists = []
        for sym in sym_permutations:  # sym: (B, N, 3)
            sym_pts = (gt_R @ sym.transpose(1, 2)).transpose(1, 2) + gt_t[:, None, :]
            dist = torch.cdist(pred_pts, sym_pts, p=2)  # (B, N, N)
            min_dist = dist.min(dim=-1)[0]              # (B, N)
            min_dists.append(min_dist.mean(dim=-1))     # list of (B,)
        add_s = torch.stack(min_dists, dim=1).min(dim=1)[0]  # (B,)

    if reduction == 'mean':
        return add_s.mean()
    return add_s

def hungarian_matching(cost_matrix):
    # Input: cost_matrix [N_gt, N_pred] (torch.Tensor, on GPU)
    import numpy as np
    import torch

    cost = cost_matrix.detach().cpu().numpy()
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    return torch.tensor(row_ind, dtype=torch.long), torch.tensor(col_ind, dtype=torch.long)