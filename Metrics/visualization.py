
import cv2
import numpy as np

render_mode = "bbox"  # Choose: "axes" or "bbox"

def draw_pose_axes(image, R, t, K, length=0.01, label="", color=(0, 255, 0)):
    axes = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    origin = np.zeros((1,3), dtype=np.float32)
    pts_3d = np.vstack((origin, axes))
    pts_2d, _ = cv2.projectPoints(pts_3d, cv2.Rodrigues(R)[0], t, K, None)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    img = image.copy()
    cv2.line(img, pts_2d[0], pts_2d[1], color, 2)
    cv2.line(img, pts_2d[0], pts_2d[2], color, 2)
    cv2.line(img, pts_2d[0], pts_2d[3], color, 2)
    return img

def draw_bbox_from_pose(image, R, t, K, size=(0.1, 0.1, 0.1), label="", color=(255, 255, 0)):
    w, h, d = size
    bbox_3d = np.array([
        [0, 0, 0], [w, 0, 0], [0, h, 0], [0, 0, d],
        [w, h, 0], [w, 0, d], [0, h, d], [w, h, d]
    ], dtype=np.float32)
    pts_2d, _ = cv2.projectPoints(bbox_3d, cv2.Rodrigues(R)[0], t, K, None)
    pts_2d = pts_2d.squeeze(1).astype(int)
    x_min, y_min = np.min(pts_2d, axis=0)
    x_max, y_max = np.max(pts_2d, axis=0)
    img = image.copy()
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    return img

