import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.CandidatePose import CandidatePoseModel
from Models.MegaPose.PoseRefiner import PoseRefiner
from Models.MegaPose.RenderBank import RenderBank
from Models.ObjectDetection.FuseNet.FuseNetFeatureEncoder import FuseNetFeatureEncoder
from Models.ObjectDetection.ResNet import ResNetFeatureEncoder

class MegaPoseEstimator(nn.Module):
    """
    A high-level MegaPose-style estimator that performs:
      1. Class-agnostic detection to generate object proposals.
      2. Coarse pose prediction (integrated into the detection head).
      3. Render-and-compare pose refinement using a bank of CAD render templates.
      
    This implementation is modeled after the MegaPose pipeline.
    """
    def __init__(self, cad_models, encoder_type="resnet", sensory_channels=None, out_dim=64,
                 renderer_params=None, distance=3.0, K=None):
        """
        Args:
            cad_models: dict mapping object id to its CAD model data.
                        Each model is expected to be a dict with keys "verts" and "faces".
            encoder_type (str): "resnet" or "fusenet" to choose the backbone.
            sensory_channels (dict): For example, {"rgb": 3}. Required for encoder initialization.
            out_dim (int): Output dimensionality for the encoder features.
            renderer_params (dict): Parameters for the KaolinRenderer.
            distance (float): Fixed camera-to-object distance for rendering.
            K: Intrinsic matrix (or list of matrices) for rendering.
        """
        super(MegaPoseEstimator, self).__init__()
        self.out_dim = out_dim

        # 1. Encoder backbone.
        if encoder_type == "resnet":
            self.encoder = ResNetFeatureEncoder(modality=list(sensory_channels.keys())[0],
                                                out_dim=self.out_dim)
        elif encoder_type == "fusenet":
            self.encoder = FuseNetFeatureEncoder(sensory_channels=sensory_channels,
                                                 ngf=self.out_dim)
        else:
            raise ValueError("Unknown encoder_type: {}".format(encoder_type))
        
        # 2. Candidate pose detection head.
        self.det_head = CandidatePoseModel(feature_dim=self.out_dim)
        
        # 3. Pose refiner network.
        self.refiner = PoseRefiner()
        
        # 4. Render bank for CAD models.
        self.render_bank = RenderBank(cad_models, renderer_params=renderer_params,
                                      distance=distance, device='cuda', K=K)
    
    def forward(self, images, candidate_cad_ids, top_k=10):
        """
        Forward pass to obtain refined pose estimations. (Used for inference.)
        
        Args:
            images: Tensor of shape (B, C, H, W).
            candidate_cad_ids: List (length B) of candidate CAD id lists for each image.
            top_k (int): Number of proposals per image.
        Returns:
            A list (length B) of pose estimation results per image,
            where each result is a dict with keys "bbox", "refined_pose", and "score".
        """
        features = self.encoder(images)
        outputs = self.det_head(features)
        proposals = self.det_head.decode_poses(outputs, top_k=top_k)  # List (B) of proposals.
        
        pose_estimations = []
        for b in range(images.size(0)):
            image_proposals = proposals[b]
            cad_ids = candidate_cad_ids[b]
            image_pose_results = []
            for proposal in image_proposals:
                real_crop = proposal["features"]  # (1, C, h, w)
                best_pose = None
                best_score = float("inf")
                for cad_id in cad_ids:
                    templates = self.render_bank.get_templates(cad_id)
                    for cad_render, render_pose in templates:
                        delta_R, delta_t = self.refiner(real_crop, cad_render)
                        refined_pose = self._combine_pose(render_pose, delta_R, delta_t)
                        score = self._compute_matching_loss(real_crop, cad_render, refined_pose)
                        if score < best_score:
                            best_score = score
                            best_pose = refined_pose
                image_pose_results.append({
                    "bbox": proposal["bbox"],
                    "refined_pose": best_pose,
                    "score": best_score
                })
            pose_estimations.append(image_pose_results)
        return pose_estimations

    def compute_loss(self, x_dict_views, K_list, cad_model_data, R_gt_list, t_gt_list, instance_id_list):
        """
        Compute the training loss for a batch.
        
        Args:
            x_dict_views: List of dictionaries mapping modality -> image tensor.
                          (Multiple views per sample.)
            K_list: Tensor of shape (V, 3, 3) for the views.
            cad_model_data: List of CAD model data tuples for each instance.
            R_gt_list: List of ground-truth rotation matrices (each shape 3x3).
            t_gt_list: List of ground-truth translation vectors (each shape 3).
            instance_id_list: List of integer instance ids corresponding to the CAD models.
        
        Returns:
            total_loss: A scalar loss tensor.
            render_loss: Loss term from render matching.
            add_s_loss: A geometric loss term (e.g. ADD-S) for the refined pose.
        """
        # For simplicity, assume we have one view per sample (or select the first view).
        # Convert the first view to a tensor.
        view = x_dict_views[0]  # Dict of modality -> tensor; assume "rgb" exists.
        if "rgb" not in view:
            raise ValueError("RGB modality is required for loss computation")
        image = view["rgb"]
        
        # Run through encoder and detection head.
        features = self.encoder(image)
        outputs = self.det_head(features)
        proposals = self.det_head.decode_poses(outputs, top_k=1)  # one proposal per image.
        
        batch_loss = 0.0
        batch_render_loss = 0.0
        batch_add_s_loss = 0.0
        
        # Process each proposal for each sample.
        for i in range(len(proposals)):
            # For each sample, assume one proposal corresponds to one GT instance.
            proposal = proposals[i][0]  # take the first proposal.
            real_crop = proposal["features"]
            cad_id = instance_id_list[i]  # candidate CAD id for this instance.
            templates = self.render_bank.get_templates(cad_id)
            
            best_loss = float("inf")
            best_refined_pose = None
            
            # For each render template of this CAD model, compute the refined pose loss.
            for cad_render, render_pose in templates:
                delta_R, delta_t = self.refiner(real_crop, cad_render)
                refined_pose = self._combine_pose(render_pose, delta_R, delta_t)
                
                # Build ground-truth pose matrix.
                gt_pose = torch.eye(4, device=refined_pose.device, dtype=refined_pose.dtype)
                gt_pose[:3, :3] = R_gt_list[i]
                gt_pose[:3, 3] = t_gt_list[i]
                
                # Compute a geometric loss between refined_pose and gt_pose.
                loss_R = F.mse_loss(refined_pose[:3, :3], gt_pose[:3, :3])
                loss_t = F.mse_loss(refined_pose[:3, 3], gt_pose[:3, 3])
                loss_instance = loss_R + loss_t
                
                # Also compute a render loss by re-rendering the CAD model using refined_pose.
                # For simplicity, we use the same cad_render here and compare with the real crop.
                cad_render_perm = cad_render.permute(0, 3, 1, 2)
                if cad_render_perm.shape[-2:] != real_crop.shape[-2:]:
                    cad_render_perm = F.interpolate(cad_render_perm, size=real_crop.shape[-2:], mode='bilinear', align_corners=False)
                render_loss = F.mse_loss(real_crop, cad_render_perm)
                
                total_instance_loss = loss_instance + render_loss
                if total_instance_loss < best_loss:
                    best_loss = total_instance_loss
                    best_refined_pose = refined_pose
            
            batch_loss += best_loss
            batch_render_loss += render_loss
            batch_add_s_loss += loss_instance
        
        N = len(proposals)
        total_loss = batch_loss / N
        render_loss_avg = batch_render_loss / N
        add_s_loss_avg = batch_add_s_loss / N
        
        return total_loss, render_loss_avg, add_s_loss_avg

    def _combine_pose(self, base_pose, delta_R, delta_t):
        """
        Compose the base pose with delta corrections.
        
        Args:
            base_pose (Tensor): Base render pose (4x4).
            delta_R (Tensor): Rotation correction (3x3).
            delta_t (Tensor): Translation correction (1x3).
        Returns:
            refined_pose (Tensor): The composed 4x4 pose matrix.
        """
        delta_pose = torch.eye(4, device=base_pose.device, dtype=base_pose.dtype)
        delta_pose[:3, :3] = delta_R
        delta_pose[:3, 3] = delta_t.squeeze(0)
        # Compose by left-multiplying the base pose.
        refined_pose = torch.matmul(delta_pose, base_pose)
        return refined_pose

    def _compute_matching_loss(self, real_crop, cad_render, refined_pose):
        """
        Compute a matching loss between the real image crop and the rendered CAD template.
        The refined_pose is used to re-render (if required) and compared against the real crop.
        
        Args:
            real_crop (Tensor): Crop from the image (1, C, h, w).
            cad_render (Tensor): Rendered CAD template (1, H, W, 3).
            refined_pose (Tensor): Final 4x4 pose.
        Returns:
            score (float): Matching loss value.
        """
        cad_render_perm = cad_render.permute(0, 3, 1, 2)
        if cad_render_perm.shape[-2:] != real_crop.shape[-2:]:
            cad_render_perm = F.interpolate(cad_render_perm, size=real_crop.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.mse_loss(real_crop, cad_render_perm)
        return loss.item()
