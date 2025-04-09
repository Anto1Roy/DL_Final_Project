
import numpy as np
import torch

def compute_add_or_adds(verts, R_gt, t_gt, R_pred, t_pred):
    # ADD metric: mean L2 distance between transformed vertices
    P_gt = (R_gt @ verts.T).T + t_gt
    P_pred = (R_pred @ verts.T).T + t_pred
    return np.linalg.norm(P_gt - P_pred, axis=1).mean()

def compute_add_gpu(verts, R_gt, t_gt, R_pred, t_pred):
    """
    Computes the ADD metric on GPU.
    
    Parameters:
        verts: Tensor of shape (V, 3) containing the 3D points (vertices) of the object.
        R_gt, R_pred: Tensors of shape (3, 3) representing the ground truth and predicted rotations.
        t_gt, t_pred: Tensors of shape (3,) representing the ground truth and predicted translations.
    
    Returns:
        add: A scalar tensor representing the average distance.
    """
    # Transform the vertices: (V, 3)
    pts_gt = torch.matmul(verts, R_gt.T) + t_gt  # Ground truth points
    pts_pred = torch.matmul(verts, R_pred.T) + t_pred  # Predicted points

    # Compute Euclidean distances per vertex and average them
    diff = pts_gt - pts_pred  # (V, 3)
    add = diff.norm(dim=1).mean()  # Scalar tensor
    return add
