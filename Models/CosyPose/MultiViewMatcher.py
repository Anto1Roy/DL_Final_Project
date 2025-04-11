import torch
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Models.helpers import quaternion_to_matrix

class MultiViewMatcher:
    def __init__(self, pose_threshold=1):
        self.pose_threshold = pose_threshold

    def pose_distance(self, R1, t1, R2, t2):
        return torch.norm(R1 - R2) + torch.norm(t1 - t2)

    def estimate_relative_pose(self, R1, t1, R2, t2):
        T1 = torch.eye(4, device=R1.device)
        T2 = torch.eye(4, device=R2.device)
        T1[:3, :3], T1[:3, 3] = R1, t1
        T2[:3, :3], T2[:3, 3] = R2, t2
        return T1 @ torch.linalg.inv(T2)

    def match(self, detections_per_view, feat_maps, K_list, extrinsics):
        """
        Matches candidate detections across views in the global coordinate frame.
        
        Args:
            detections_per_view: List of detections per view.
            feat_maps: List of feature maps per view.
            K_list: List of camera intrinsics (one per view).
            extrinsics: List of dictionaries (one per view), each with keys:
                        "R_w2c" (Tensor of shape (3,3)) and "t_w2c" (Tensor of shape (3,))
        
        Returns:
            A list of clusters (each a list of detections) that are matched across views.
        """
        all_detections = []
        for view_idx, detections in enumerate(detections_per_view):
            ext = extrinsics[view_idx]
            R_w2c = ext["R_w2c"] 
            t_w2c = ext["t_w2c"]
            for det in detections:
                R_c = quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
                t_c = det['trans']
                # Get global pose in world coordinates.
                R_global = torch.matmul(R_w2c.T, R_c)
                t_global = torch.matmul(R_w2c.T, (t_c - t_w2c))
                all_detections.append({
                    'quat': det['quat'],
                    'trans': det['trans'],
                    'R': R_global,      
                    'view_idx': view_idx,
                    'score': det['score'],
                    'feat': feat_maps[view_idx],
                    'K': K_list[view_idx],
                    'R_global': R_global,
                    't_global': t_global
                })

        n = len(all_detections)
        if n == 0:
            return []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d1, d2 = all_detections[i], all_detections[j]
                if d1['view_idx'] == d2['view_idx']:
                    continue  # Skip detections from the same view
                dist = self.pose_distance(d1['R_global'], d1['t_global'],
                                          d2['R_global'], d2['t_global'])
                if dist < self.pose_threshold:
                    edges.append((i, j))

        if len(edges) > 0:
            rows, cols = zip(*edges)
            graph = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
            n_components, labels = connected_components(csgraph=graph, directed=False)
            
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                clusters[label].append(all_detections[idx])
            return list(clusters.values())
        else:
            print("[INFO] No matching detections across views — using fallback per-view grouping.")
            clusters = [[d] for d in all_detections]
            return clusters
