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

    def match(self, detections_per_view, feat_maps, K_list):
        all_detections = []
        for view_idx, detections in enumerate(detections_per_view):
            for det in detections:
                R = quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
                all_detections.append({
                    'quat': det['quat'],
                    'trans': det['trans'],
                    'R': R,
                    'view_idx': view_idx,
                    'score': det['score'],
                    'feat': feat_maps[view_idx],
                    'K': K_list[view_idx]
                })

        n = len(all_detections)

        if n == 0:
            return []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d1, d2 = all_detections[i], all_detections[j]
                if d1['view_idx'] == d2['view_idx']:
                    continue  # skip same view

                dist = self.pose_distance(d1['R'], d1['trans'], d2['R'], d2['trans'])
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
            # Fallback: treat each detection as its own group
            print("[INFO] No matching detections across views — using fallback per-view grouping.")
            clusters = [[d] for d in all_detections]
            return clusters