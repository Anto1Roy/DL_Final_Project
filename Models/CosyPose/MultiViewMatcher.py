import torch
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Models.helpers import quaternion_to_matrix

def average_quaternions(quats):
    """Averages a set of quaternions using the eigenvector method."""
    A = torch.zeros((4, 4), device=quats.device)
    for q in quats:
        A += torch.outer(q, q)
    eigvals, eigvecs = torch.linalg.eigh(A)
    return eigvecs[:, -1]

class MultiViewMatcher:
    def __init__(self, pose_threshold=1):
        self.pose_threshold = pose_threshold

    def pose_distance(self, R1, t1, R2, t2):
        return torch.norm(R1 - R2) + torch.norm(t1 - t2)

    def match(self, detections_per_view, K_list, extrinsics):
        all_detections = []
        for view_idx, detections in enumerate(detections_per_view):
            ext = extrinsics[view_idx]
            R_w2c, t_w2c = ext["R_w2c"], ext["t_w2c"]

            for det in detections:
                R_c = quaternion_to_matrix(det['quat'].unsqueeze(0))[0]
                t_c = det['trans']
                R_global = torch.matmul(R_w2c.T, R_c)
                t_global = torch.matmul(R_w2c.T, (t_c - t_w2c))

                all_detections.append({
                    'quat': det['quat'],
                    'trans': det['trans'],
                    'R_global': R_global,
                    't_global': t_global,
                    'score': det['score'],
                    'K': K_list[view_idx],
                    'cam_extrinsics': ext,
                    'view_idx': view_idx
                })

        n = len(all_detections)
        if n == 0:
            return []

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d1, d2 = all_detections[i], all_detections[j]
                if d1['view_idx'] == d2['view_idx']:
                    continue
                dist = self.pose_distance(d1['R_global'], d1['t_global'],
                                          d2['R_global'], d2['t_global'])
                if dist < self.pose_threshold:
                    edges.append((i, j))

        if edges:
            rows, cols = zip(*edges)
            graph = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
            n_components, labels = connected_components(csgraph=graph, directed=False)
        else:
            print("[INFO] No matching detections across views — using fallback per-view grouping.")
            labels = np.arange(n)
            n_components = n

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(all_detections[idx])

        merged = []
        for instance_id, cluster in clusters.items():
            quats = torch.stack([d['quat'] for d in cluster])
            trans = torch.stack([d['t_global'] for d in cluster])
            feats = torch.stack([d['feat'].mean(dim=[2, 3]) for d in cluster])

            avg_quat = average_quaternions(quats)
            avg_trans = trans.mean(dim=0)
            avg_feat = feats.mean(dim=0).unsqueeze(-1).unsqueeze(-1)  # Cx1x1

            merged.append({
                'instance_id': instance_id,
                'quat': avg_quat,
                'trans': avg_trans,
                'K': cluster[0]['K'],
                'cam_extrinsics': self._to_extrinsics_matrix(cluster[0]['cam_extrinsics'])
            })

        return merged

    def _to_extrinsics_matrix(self, ext_dict):
        R_w2c = ext_dict['R_w2c']
        t_w2c = ext_dict['t_w2c']
        T = torch.eye(4, device=R_w2c.device)
        T[:3, :3] = R_w2c
        T[:3, 3] = t_w2c
        return T
