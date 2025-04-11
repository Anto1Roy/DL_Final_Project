import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from Models.CosyPose.CandidatePose import CandidatePoseModel
from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.Encoding.ResNet34 import ResNetFeatureEncoder
from Models.PoseEstimator.TransformerFusion import TransformerFusion
from Models.PoseEstimator.RenderAndEmbed import RenderAndEmbed
from Models.helpers import quaternion_to_matrix
from Models.KaolinRenderer import KaolinRenderer  


class PoseEstimator(nn.Module):
    def __init__(
        self,
        sensory_channels,
        renderer_config,
        encoder_type="resnet",
        n_views=1,
        fusion_type=None,
        num_candidates=32,
        noise_level=0.05,
        conf_thresh=0.5,
        lambda_desc=10.0,
        device="cuda"
    ):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh
        self.out_dim = 16
        self.fusion_type = fusion_type
        self.lambda_desc = lambda_desc
        self.device = device

        # Feature encoder.
        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.out_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Optional view fusion (Transformer).
        if fusion_type == "transformer":
            self.fusion_module = TransformerFusion(in_channels=self.out_dim, hidden_dim=64, num_heads=4, num_layers=2)
            self.det_head = CandidatePoseModel(feature_dim=self.fusion_module.out_dim)
        else:
            self.fusion_module = None
            self.det_head = CandidatePoseModel(feature_dim=self.out_dim * n_views)

        # CAD Embedding module (RenderAndEmbed).
        renderer = KaolinRenderer(**renderer_config)
        self.embed_model = RenderAndEmbed(renderer=renderer, output_dim=128)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect(self, feat_map, class_embeddings, top_k=100, score_thresh=0.5, extrinsics=None):
        # Pass extrinsics (and intrinsics if needed) to the candidate detector.
        outputs = self.det_head.forward(feat_map, K=None, extrinsics=extrinsics)
        # Decode detections for the first sample in the batch.
        detections_batch = self.det_head.match_and_decode(outputs, class_embeddings, top_k=top_k)[0]
        # Filter out detections with a score below the threshold.
        detections_batch_filtered = [det for det in detections_batch if det['score'] >= score_thresh]
        return detections_batch_filtered, outputs

    def forward(self, x_dict_views, K_list, cad_model_lookup, extrinsics, top_k=100):
        # Compute class embeddings for each CAD object.
        class_embeddings = {
            obj_id: self.embed_model(
                item["verts"], item["faces"],
                R=torch.eye(3, device=self.device),
                T=torch.zeros(3, device=self.device),
                K=K_list[0] if K_list is not None else None
            )
            for obj_id, item in cad_model_lookup.items()
        }
        # Extract per-view features.
        feat_maps = [self.extract_single_view_features(x) for x in x_dict_views]
        # If fusion is available, use it; otherwise, concatenate features.
        if self.fusion_module:
            fused_map = self.fusion_module(feat_maps, K_list, extrinsics=extrinsics)
        else:
            fused_map = torch.cat(feat_maps, dim=1)
        # Use extrinsics in the detection branch.
        detections, outputs = self.detect(fused_map, class_embeddings, top_k=top_k, extrinsics=extrinsics)
        return detections, outputs, class_embeddings

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, K_list, cad_model_lookup, extrinsics, bbox_gt_list):
        """
        Computes the pose loss by comparing predicted candidate poses (transformed into global coordinates)
        to the ground-truth poses (also transformed into global coordinates). Additionally, a multi-object 
        descriptor loss is computed.
        
        For each view, both the predicted translation and the GT translation are adjusted using the 
        corresponding camera intrinsics (here, via simple normalization by focal lengths) and then transformed 
        to global coordinates using the extrinsics.
        
        Args:
            x_dict_views: List of view dictionaries {modality -> Tensor(C, H, W)}.
            R_gt_list: Nested list over views; per view, a list of GT rotation matrices (in camera coordinates).
            t_gt_list: Nested list over views; per view, a list of GT translation vectors (in camera coordinates).
            K_list: List (or Tensor) of camera intrinsics (one per view; each a tensor of shape (3,3)).
            cad_model_lookup: Dictionary mapping object IDs to CAD model data.
            extrinsics: List of dictionaries (one per view), each with keys "R_w2c" (Tensor(3,3)) and "t_w2c" (Tensor(3)).
        
        Returns:
            total_loss, avg_rot, avg_trans, desc_loss
        """
        device = K_list[0].device

        total_rot_loss = 0.0
        total_trans_loss = 0.0
        total_class_loss = 0.0
        total_conf_loss = 0.0
        total_bbox_loss = 0.0
        total_matches = 0

        num_views = len(x_dict_views)

        # Process each view independently.
        for view_idx in range(num_views):
            # --- Extract features and detections for current view ---
            feat = self.extract_single_view_features(x_dict_views[view_idx])
            outputs = self.det_head.forward(feat, K=K_list[view_idx], extrinsics=extrinsics[view_idx])
            outputs = self.det_head.apply_activations(outputs)
            # Auxiliary bounding-box prediction.
            bbox_pred = self.det_head.bbox_head(feat)  # [1, 4, H, W] (if available from the candidate model)
            _, _, H, W = bbox_pred.shape

            view_bbox_loss = 0.0
            # Assume bbox_gt_list[view_idx] is a dict with key "bbox_visib_list"
            if (bbox_gt_list is not None) and (view_idx < len(bbox_gt_list)):
                bbox_gt_view = bbox_gt_list[view_idx]
                pred_bbox_map = bbox_pred[0].permute(1, 2, 0)  # [H, W, 4]
                center_h = H // 2
                center_w = W // 2
                pred_box = pred_bbox_map[center_h, center_w]
                if "bbox_visib_list" in bbox_gt_view and len(bbox_gt_view["bbox_visib_list"]) > 0:
                    gt_box = torch.tensor(bbox_gt_view["bbox_visib_list"][0], dtype=torch.float32, device=device)
                    view_bbox_loss += F.l1_loss(pred_box, gt_box)
            total_bbox_loss += view_bbox_loss

            # --- Process pose predictions for current view ---
            # Assume batch size is 1.
            quat = outputs['quat'][0].reshape(-1, 4)
            trans = outputs['trans'][0].reshape(-1, 3)
            conf = outputs['conf'][0].reshape(-1)
            class_scores = outputs['class_scores'][0].reshape(-1, outputs['class_scores'].shape[-1])
            pred_classes = class_scores.argmax(dim=-1)
            pred_Rs = quaternion_to_matrix(quat)

            # Get current view extrinsics and intrinsics.
            R_w2c = extrinsics[view_idx]["R_w2c"].to(device)
            t_w2c = extrinsics[view_idx]["t_w2c"].to(device)
            K = K_list[view_idx].to(device)
            # Compute a simple normalization factor from the intrinsics (using fx and fy).
            fx = K[0, 0]
            fy = K[1, 1]
            scale = torch.tensor([1.0/fx, 1.0/fy, 1.0], device=device)

            # Transform GT poses for current view into global coordinates.
            view_gt_Rs = []
            view_gt_ts = []
            for i, obj_id in enumerate(R_gt_list[view_idx]):
                R_gt = R_gt_list[view_idx][i].to(device)
                t_gt = t_gt_list[view_idx][i].to(device)
                # If t_gt is provided in pixel coordinates with depth, apply intrinsic normalization.
                t_gt_norm = t_gt * scale  # (3,) scale adjustment as a simple example.
                R_gt_global = torch.matmul(R_w2c.T, R_gt)
                t_gt_global = torch.matmul(R_w2c.T, (t_gt_norm - t_w2c).unsqueeze(-1)).squeeze(-1)
                view_gt_Rs.append(R_gt_global)
                view_gt_ts.append(t_gt_global)
            if len(view_gt_Rs) == 0:
                continue
            view_gt_Rs = torch.stack(view_gt_Rs)  # (N_gt, 3, 3)
            view_gt_ts = torch.stack(view_gt_ts)    # (N_gt, 3)

            # Transform predicted poses for current view into global coordinates.
            pred_R_global = []
            pred_t_global = []
            for j in range(quat.shape[0]):
                R_cam = quaternion_to_matrix(quat[j].unsqueeze(0))[0]
                t_cam = trans[j]
                # Optionally, you might normalize predicted translation using K as well:
                t_cam_norm = t_cam * scale
                R_pred_global = torch.matmul(R_w2c.T, R_cam)
                t_pred_global = torch.matmul(R_w2c.T, (t_cam_norm - t_w2c).unsqueeze(-1)).squeeze(-1)
                pred_R_global.append(R_pred_global)
                pred_t_global.append(t_pred_global)
            if len(pred_R_global) == 0:
                continue
            pred_R_global = torch.stack(pred_R_global)
            pred_t_global = torch.stack(pred_t_global)

            # --- Matching for current view ---
            rot_diff = torch.cdist(view_gt_Rs.view(len(view_gt_Rs), -1), pred_R_global.view(len(pred_R_global), -1))
            trans_diff = torch.cdist(view_gt_ts.view(len(view_gt_ts), -1), pred_t_global.view(len(pred_t_global), -1))
            cost_matrix = (rot_diff + trans_diff).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                total_rot_loss += F.mse_loss(pred_R_global[j], view_gt_Rs[i])
                total_trans_loss += F.mse_loss(pred_t_global[j], view_gt_ts[i])
                # Classification and confidence losses are placeholders.
                dummy_class = torch.tensor([0], device=device)
                total_class_loss += F.cross_entropy(class_scores[j].unsqueeze(0), dummy_class)
                total_conf_loss += F.binary_cross_entropy_with_logits(conf[j].unsqueeze(0),
                                                                       conf[j].unsqueeze(0))
                total_matches += 1

        if total_matches == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero, zero, zero, zero

        avg_rot = total_rot_loss / total_matches
        avg_trans = total_trans_loss / total_matches
        avg_class = total_class_loss / total_matches
        avg_conf = total_conf_loss / (total_matches + 1e-6)
        avg_bbox = total_bbox_loss / num_views

        total_loss = avg_rot + avg_trans + avg_class + avg_conf + 0.1 * avg_bbox

        # Additionally compute the multi-object descriptor loss.
        pred_embed = outputs["embed"]  # [B, D, H, W]
        desc_losses = []
        for obj_id, obj_embed in class_embeddings.items():
            gt_embedding = obj_embed.view(1, -1, 1, 1)  # [1, D, 1, 1]
            sim_map = F.cosine_similarity(pred_embed, gt_embedding, dim=1)  # [B, H, W]
            topk_sim, _ = sim_map.view(sim_map.shape[0], -1).topk(100, dim=1)
            desc_loss_obj = 1 - topk_sim.mean()
            desc_losses.append(desc_loss_obj)
        if len(desc_losses) > 0:
            desc_loss = torch.stack(desc_losses).mean()
        else:
            desc_loss = torch.tensor(0.0, device=device)
        total_loss = total_loss + self.lambda_desc * desc_loss

        return total_loss, avg_rot, avg_trans, avg_class, avg_conf, avg_bbox
