import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from Models.MultiViewMatcher import MultiViewMatcher
from Models.CozyPose import CosyPoseStyleRenderMatch
from Models.CandidatePose import CandidatePoseModel
from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder
from Models.SceneRefinder import SceneRefiner
from Models.helpers import quaternion_to_matrix

def pose_distance(R1, t1, R2, t2):
    rot_diff = torch.norm(R1 - R2)
    trans_diff = torch.norm(t1 - t2)
    return rot_diff + trans_diff


class TwoStagePoseEstimator(nn.Module):
    def __init__(self, sensory_channels, renderer_config, encoder_type="resnet", num_candidates=32, noise_level=0.05, conf_thresh=0.5):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh
        self.out_dim = 16

        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.out_dim)

        self.det_head = CandidatePoseModel(feature_dim=self.out_dim)
        self.refiner = CosyPoseStyleRenderMatch(renderer_config, num_candidates=num_candidates)
        self.matcher = MultiViewMatcher(pose_threshold=0.1)
        self.global_refiner = SceneRefiner(renderer_config)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect_single_view(self, feat_map, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections_per_sample = self.det_head.decode_poses(outputs, top_k=top_k)
        filtered = [
            [d for d in dets if d["score"] >= self.conf_thresh]
            for dets in detections_per_sample
        ]
        return filtered, outputs

    def sample_candidates(self, detections):
        candidates_all = []
        for det in detections:
            quat = det["quat"].unsqueeze(0).expand(self.num_candidates, -1)
            trans = det["trans"].unsqueeze(0).expand(self.num_candidates, -1)
            noise_q = torch.randn_like(quat) * self.noise_level
            noise_t = torch.randn_like(trans) * self.noise_level
            noisy_q = F.normalize(quat + noise_q, dim=-1)
            noisy_t = trans + noise_t
            R = quaternion_to_matrix(noisy_q)
            poses = torch.eye(4, device=quat.device).repeat(self.num_candidates, 1, 1)
            poses[:, :3, :3] = R
            poses[:, :3, 3] = noisy_t
            candidates_all.append(poses)
        return candidates_all

    def match_gt_with_detections_across_views(self, detections_per_view, R_gt_tensor, t_gt_tensor, instance_id_list, sample_indices, feat_maps, Ks):
        matched_detections = []
        matched_gt_R, matched_gt_t, matched_ids, matched_feat, matched_K = [], [], [], [], []

        objects_by_id = defaultdict(list)
        for i in range(len(sample_indices)):
            sample_idx = sample_indices[i].item()
            obj_id = instance_id_list[i]
            objects_by_id[obj_id].append((sample_idx, R_gt_tensor[i], t_gt_tensor[i]))

        for obj_id, gt_info_list in objects_by_id.items():
            for sample_idx, R_gt, t_gt in gt_info_list:
                dets = detections_per_view[sample_idx]
                best_match, best_dist = None, float('inf')
                for det in dets:
                    quat = det["quat"]
                    t_pred = det["trans"]
                    R_pred = quaternion_to_matrix(quat.unsqueeze(0))[0]
                    dist = torch.norm(R_gt - R_pred) + torch.norm(t_gt - t_pred)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = det
                if best_match is not None:
                    matched_detections.append(best_match)
                    matched_gt_R.append(R_gt)
                    matched_gt_t.append(t_gt)
                    matched_ids.append(obj_id)
                    matched_feat.append(feat_maps[sample_idx])
                    matched_K.append(Ks[sample_idx])

        return matched_detections, matched_gt_R, matched_gt_t, matched_ids, matched_feat, matched_K

    def compute_loss(self, x_dict_views, K_list, cad_model_data, R_gt_list, t_gt_list, instance_id_list):
        """
        Args:
            x_dict_views: List[Dict[modality -> Tensor(C, H, W)]]
            K_list: Tensor of shape (V, 3, 3)
            cad_model_data: List of (verts, faces) per object
            R_gt_list: List[Tensor(3, 3)]
            t_gt_list: List[Tensor(3)]
            instance_id_list: List[int]
        """

        # Stage 1: Feature extraction per view
        feat_maps = [
            self.extract_single_view_features(
                {mod: view[mod].unsqueeze(0) for mod in view}
            ) for view in x_dict_views
        ]

        # Stage 2: Detection
        detections_per_view = [self.detect_single_view(fm, top_k=25)[0][0] for fm in feat_maps]

        # Stage 3: Multi-view hypothesis matching
        object_groups = self.matcher.match(detections_per_view, feat_maps, K_list)

        # Stage 4: Refiner
        total_loss, render_losses, add_s = self.global_refiner.compute_loss(
            object_groups,
            cad_model_lookup={obj_id: cad_model_data[i] for i, obj_id in enumerate(instance_id_list)},
            gt_R_tensor=torch.stack(R_gt_list) if R_gt_list else None,
            gt_t_tensor=torch.stack(t_gt_list) if t_gt_list else None,
            instance_id_list=instance_id_list
        )

        return total_loss, render_losses, add_s
    
    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, K_list):
        device = K_list[0].device

        # Extract features per view
        feat_maps = [
            self.extract_single_view_features(
                {mod: view[mod].unsqueeze(0) for mod in view}
            ) for view in x_dict_views
        ]

        # Stack all GT poses across views (they're the same for all views)
        gt_Rs = torch.stack(R_gt_list).to(device)  # shape: (num_gt, 3, 3)
        gt_ts = torch.stack(t_gt_list).to(device)  # shape: (num_gt, 3)

        all_pred_Rs = []
        all_pred_ts = []

        # Collect all detections from all views
        for i, feat_map in enumerate(feat_maps):
            detections_batch, _ = self.detect_single_view(feat_map, top_k=5)
            detections = detections_batch[0]

            if len(detections) == 0:
                continue

            pred_Rs = torch.stack([
                quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
                for det in detections
            ])
            pred_ts = torch.stack([
                det['trans'] for det in detections
            ])

            all_pred_Rs.append(pred_Rs)
            all_pred_ts.append(pred_ts)

        if len(all_pred_Rs) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        # Combine all detections from all views
        all_pred_Rs = torch.cat(all_pred_Rs, dim=0)  # shape: (N_pred, 3, 3)
        all_pred_ts = torch.cat(all_pred_ts, dim=0)  # shape: (N_pred, 3)

        num_gt = len(gt_Rs)
        num_pred = len(all_pred_Rs)

        # Compute pairwise distances
        rot_diff = torch.cdist(gt_Rs.view(num_gt, -1), all_pred_Rs.view(num_pred, -1))
        trans_diff = torch.cdist(gt_ts.view(num_gt, -1), all_pred_ts.view(num_pred, -1))

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost_matrix = (rot_diff + trans_diff).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        losses_rot = []
        losses_trans = []

        for gt_idx, pred_idx in zip(row_ind, col_ind):
            losses_rot.append(rot_diff[gt_idx, pred_idx])
            losses_trans.append(trans_diff[gt_idx, pred_idx])

        losses_rot = torch.stack(losses_rot)
        losses_trans = torch.stack(losses_trans)
        total_loss = losses_rot.mean() + losses_trans.mean()

        return total_loss, losses_rot.mean(), losses_trans.mean()

        

    def forward(self, x_dict_list, K_list, cad_model_lookup, top_k=100):
        # Stage 1: extract features + detections
        feat_maps = [self.extract_single_view_features(x_dict) for x_dict in x_dict_list]
        detections_per_view = [self.detect_single_view(f, top_k=top_k)[0][0] for f in feat_maps]

        print("Detections per view:")
        for i, dets in enumerate(detections_per_view):
            print(f"View {i}: {len(dets)} detections")

        # Stage 2: match multi-view object hypotheses
        object_groups = self.matcher.match(detections_per_view, feat_maps, K_list)

        print("Object groups after matching:")
        for i, group in enumerate(object_groups):
            print(f"Group {i}: {len(group)} detections")

        # Stage 3: refine the scene globally
        refined_poses, rendered_views = self.global_refiner(object_groups, cad_model_lookup)

        print("Refined poses and rendered views:")
        for i, (pose, view) in enumerate(zip(refined_poses, rendered_views)):
            print(f"Pose {i}: {pose}")
            print(f"View {i}: {view}")

        return object_groups, refined_poses, rendered_views 

    def freeze_refiner(self):
        for param in self.refiner.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.det_head.parameters():
            param.requires_grad = False