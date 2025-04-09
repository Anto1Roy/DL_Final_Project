import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from Models.helpers import hungarian_matching, quaternion_to_matrix, quaternion_to_matrix_no_B, rotation_to_quat
from scipy.optimize import linear_sum_assignment
from einops import rearrange
from torchvision.models import ResNet18_Weights


def quaternion_distance(q1, q2):
    # Computes angular distance in radians between two quaternions
    dot = torch.sum(q1 * q2, dim=-1).clamp(-1, 1)
    angle = 2 * torch.acos(dot.abs())
    return angle
def rotation_distance(R1, R2):
    # R1, R2: (..., 3, 3) → angular difference
    trace = (R1.transpose(-1, -2) @ R2).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    return torch.acos(((trace - 1) / 2).clamp(-1, 1))  # radians

def translation_distance(t1, t2):
    return torch.norm(t1 - t2, dim=-1)

class PoseEstimator(nn.Module):
    def __init__(self, obj_ids, n_views, backbone_name='resnet18', fusion='concat'):
        super().__init__()
        self.obj_ids = sorted(list(obj_ids))  # Ensure consistent ordering
        self.obj_id_to_class_idx = {obj_id: idx for idx, obj_id in enumerate(self.obj_ids)}
        self.class_idx_to_obj_id = {idx: obj_id for obj_id, idx in self.obj_id_to_class_idx.items()}
        self.fusion_type = fusion

        # Pretrained encoder
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # (B, 512, H/32, W/32)
        )
        # self.encoder.fc = nn.Identity()  # remove classification head

        # Output: (B, 512, H/32, W/32)
        self.detector = nn.Sequential(
            nn.Conv2d(n_views * 512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, (4 + 3 + len(self.obj_ids) + 1), 1)  # quat(4) + t(3) + class + conf
        )

    def encode_view(self, image):
        image = image.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
        return self.encoder(image)

    def fuse_views(self, features_list):
        return torch.cat(features_list, dim=1)  # (B, V*C, H, W)

    def forward(self, x_dict_views):
        feats = [self.encode_view(view['rgb']) for view in x_dict_views]  # list of tensors
        fused = self.fuse_views(feats)
        out = self.detector(fused)
        return out

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, gt_obj_ids, K_list):

        pred = self.forward(x_dict_views)

        # Decode predictions
        poses_pred, obj_ids_pred, conf_pred = self.decode_predictions(pred)

        # Match predicted poses to GT using Hungarian or greedy matching
        matched = self.match_predictions_to_gt(poses_pred, obj_ids_pred, gt_obj_ids, R_gt_list, t_gt_list)

        # Compute loss
        rot_loss, trans_loss, class_loss, conf_loss = self.compute_losses(matched, K_list)

        total = rot_loss, trans_loss, class_loss, conf_loss
        return total, rot_loss, trans_loss, class_loss, conf_loss

    def decode_predictions(self, pred):
        # pred: (B, D, H, W)
        B, D, H, W = pred.shape
        pred = rearrange(pred, 'b d h w -> b h w d')
        quat = F.normalize(pred[..., :4], dim=-1)
        trans = pred[..., 4:7]
        class_logits = pred[..., 7:-1]
        conf = torch.sigmoid(pred[..., -1])
        class_idx = class_logits.argmax(-1)  # (B, H, W)
        obj_ids = class_idx.clone()  # will be replaced with real obj ids

        # Vectorized mapping (convert class index to obj ID using lookup table)
        lookup_table = torch.tensor([self.class_idx_to_obj_id[i] for i in range(len(self.obj_ids))], device=class_idx.device)
        obj_ids = lookup_table[class_idx]  # (B, H, W) with real object IDs

        B, H, W, _ = quat.shape
        quat = quat.view(-1, 4)         # (B*H*W, 4)
        trans = trans.view(-1, 3)       # (B*H*W, 3)
        obj_ids = obj_ids.view(-1)      # (B*H*W,)
        conf = conf.view(-1)            # (B*H*W,)

        return (quat, trans), obj_ids, conf

    def match_predictions_to_gt(self, poses_pred, obj_ids_pred, gt_obj_ids, R_gt_list, t_gt_list):
        quat_pred, trans_pred = poses_pred  # (Np, 4), (Np, 3)
        obj_ids_pred = obj_ids_pred         # (Np,)
        gt_obj_ids = torch.tensor(gt_obj_ids, device=trans_pred.device)  # (Ng,)
        R_gt = torch.stack(R_gt_list)       # (Ng, 3, 3)
        t_gt = torch.stack(t_gt_list)       # (Ng, 3)

        Np = quat_pred.shape[0]
        Ng = R_gt.shape[0]

        quat_pred_flat = quat_pred.reshape(-1, 4)  # (B * H * W, 4)
        R_pred = quaternion_to_matrix_no_B(quat_pred_flat)  # (B * H * W, 3, 3)

        # Prepare cost matrix (Np, Ng) — high cost for object ID mismatch
        cost_matrix = torch.full((Np, Ng), fill_value=1e6, device=quat_pred.device)

        for i in range(Np):
            for j in range(Ng):
                if obj_ids_pred[i] == gt_obj_ids[j]:
                    rot_dist = rotation_distance(R_pred[i], R_gt[j])
                    trans_dist = translation_distance(trans_pred[i], t_gt[j])
                    cost_matrix[i, j] = rot_dist + trans_dist  # Weighted sum

        # Apply Hungarian matching on cost_matrix (lower = better)
        row_ind, col_ind = hungarian_matching(cost_matrix)

        # Filter invalid matches (where cost was infinite)
        valid = cost_matrix[row_ind, col_ind] < 1e5
        row_ind = torch.tensor(row_ind, device=quat_pred.device)[valid]
        col_ind = torch.tensor(col_ind, device=quat_pred.device)[valid]

        # Select matched predictions and GT
        quat_pred_matched = quat_pred[row_ind]
        trans_pred_matched = trans_pred[row_ind]
        obj_ids_pred_matched = obj_ids_pred[row_ind]

        R_gt_matched = R_gt[col_ind]
        t_gt_matched = t_gt[col_ind]
        gt_obj_ids_matched = gt_obj_ids[col_ind]

        return {
            "quat_pred": quat_pred_matched,
            "trans_pred": trans_pred_matched,
            "obj_ids_pred": obj_ids_pred_matched,
            "R_gt": R_gt_matched,
            "t_gt": t_gt_matched,
            "gt_obj_ids": gt_obj_ids_matched,
            "conf_pred": torch.ones_like(obj_ids_pred_matched).float()  # optional
        }

    def compute_losses(self, matched, K_list=None):
        # Unpack matched
        quat_pred = matched["quat_pred"]           # (N, 4)
        trans_pred = matched["trans_pred"]         # (N, 3)
        obj_ids_pred = matched["obj_ids_pred"]     # (N,)
        R_gt = matched["R_gt"]                     # (N, 3, 3)
        t_gt = matched["t_gt"]                     # (N, 3)
        gt_obj_ids = matched["gt_obj_ids"]         # (N,)
        class_logits = matched["class_logits"]
        conf_pred = matched.get("conf_pred", torch.ones_like(obj_ids_pred, dtype=torch.float32))  # (N,)
        conf_gt = torch.ones_like(conf_pred)       # All matched → confidence = 1

        device = quat_pred.device

        # === Rotation Loss ===
        # R_pred = quaternion_to_matrix_no_B(quat_pred)               # (N, 3, 3)
        q_gt = rotation_to_quat(R_gt)        # (N, 4)
        rot_loss = quaternion_distance(quat_pred, q_gt).mean() # scalar

        # === Translation Loss ===
        trans_loss = F.mse_loss(trans_pred, t_gt)

        # === Classification Loss ===
        gt_labels = torch.tensor(
            [self.obj_id_to_class_idx[int(obj_id)] for obj_id in gt_obj_ids],
            device=gt_obj_ids.device
        )

        class_loss = F.cross_entropy(class_logits, gt_labels)

        # === Confidence Loss ===
        conf_loss = F.binary_cross_entropy(conf_pred.float(), conf_gt.float())

        return rot_loss, trans_loss, class_loss, conf_loss
