import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from Models.CosyPose.CandidatePose import CandidatePoseModel
from Models.ObjectDetection.FuseNet.FuseNet import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder
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
        conf_thresh=0.5
    ):
        super().__init__()
        self.num_candidates = num_candidates
        self.noise_level = noise_level
        self.conf_thresh = conf_thresh
        self.out_dim = 16
        self.fusion_type = fusion_type

        # Feature encoder
        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0], out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels, ngf=self.out_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Optional view fusion (Transformer)
        if fusion_type == "transformer":
            self.fusion_module = TransformerFusion(in_channels=self.out_dim, hidden_dim=64, num_heads=4, num_layers=2)
            self.det_head = CandidatePoseModel(feature_dim=self.fusion_module.out_dim)
        else:
            self.fusion_module = None
            self.det_head = CandidatePoseModel(feature_dim=self.out_dim * n_views)

        # CAD Embedding module (RenderAndEmbed)
        renderer = KaolinRenderer(**renderer_config)
        self.embed_model = RenderAndEmbed(renderer=renderer, output_dim=128)

    def extract_single_view_features(self, x_dict):
        return self.encoder(x_dict)

    def detect(self, feat_map, class_embeddings, top_k=15):
        outputs = self.det_head.forward(feat_map)
        detections_batch = self.det_head.match_and_decode(outputs, class_embeddings, top_k=top_k)[0]
        return detections_batch, outputs

    def forward(self, x_dict_views, K_list=None, cad_model_lookup=None, top_k=15):
        feat_maps = [self.extract_single_view_features(x) for x in x_dict_views]

        if self.fusion_module:
            fused_map = self.fusion_module(feat_maps)
        else:
            fused_map = torch.cat(feat_maps, dim=1)

        detections, _ = self.detect(fused_map, cad_model_lookup, top_k=top_k)
        return detections

    def compute_pose_loss(self, x_dict_views, R_gt_list, t_gt_list, K_list, cad_model_lookup):
        device = K_list[0].device

        class_embeddings = {
            obj_id: self.embed_model(
                item["verts"], item["faces"],
                R=torch.eye(3, device=device),              # (3, 3)
                T=torch.zeros(3, device=device),            # (3,)
                K=K_list[0] if K_list is not None else None # (3, 3)
            )
            for obj_id, item in cad_model_lookup.items()
        }

        # Forward pass
        detections_batch = self.forward(x_dict_views, K_list=K_list, cad_model_lookup=class_embeddings, top_k=15)

        if len(detections_batch) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        # Predicted poses
        pred_Rs = torch.stack([
            quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
            for det in detections_batch
        ])
        pred_ts = torch.stack([det['trans'] for det in detections_batch])

        # Ground truth poses
        gt_Rs = torch.stack(R_gt_list).to(device)
        gt_ts = torch.stack(t_gt_list).to(device)

        # Matching predictions to ground truth using Hungarian algorithm
        rot_diff = torch.cdist(gt_Rs.view(len(gt_Rs), -1), pred_Rs.view(len(pred_Rs), -1))
        trans_diff = torch.cdist(gt_ts, pred_ts)
        cost_matrix = (rot_diff + trans_diff).detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Final pose losses
        losses_rot = torch.stack([rot_diff[i, j] for i, j in zip(row_ind, col_ind)])
        losses_trans = torch.stack([trans_diff[i, j] for i, j in zip(row_ind, col_ind)])
        total_loss = losses_rot.mean() + losses_trans.mean()

        desc_losses = []

        for obj_id, obj_embed in class_embeddings.items():
            gt_embedding = obj_embed.view(1, -1, 1, 1)  # [1, D, 1, 1]
            similarity_map = F.cosine_similarity(pred_embed, gt_embedding, dim=1)  # [B, H, W]
            
            topk_sim, _ = similarity_map.view(similarity_map.shape[0], -1).topk(500, dim=1)
            desc_loss_obj = 1 - topk_sim.mean()
            desc_losses.append(desc_loss_obj)

        desc_loss = torch.stack(desc_losses).mean()
        total_loss = pose_loss + self.lambda_desc * desc_loss

        return total_loss, losses_rot.mean(), losses_trans.mean()