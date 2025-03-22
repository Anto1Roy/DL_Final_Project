# File Format Specification for BOP-style Camera and Ground Truth Files

This document describes the expected format for files used in the multi-camera 6D pose estimation pipeline. Each `X` refers to a camera index (e.g., `scene_camera_cam1.json`, ..., `scene_camera_cam3.json`).

---

## ✅ scene_camera_camX.json

Contains intrinsic and extrinsic camera parameters per frame.

```json
{
  "0": {
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "depth_scale": 0.001,
    "cam_R_w2c": [r11, r12, r13, r21, ..., r33],
    "cam_t_w2c": [tx, ty, tz]
  },
  "1": {
    ...
  }
}
```

- `cam_K`: 3x3 camera intrinsic matrix in row-major format
- `depth_scale`: scale to convert depth images to meters
- `cam_R_w2c`: rotation matrix from world to camera (3x3)
- `cam_t_w2c`: translation vector from world to camera (3,)

---

## ✅ scene_gt_camX.json

Provides ground truth object poses for each frame.

```json
{
  "0": [
    {
      "cam_R_m2c": [r11, r12, r13, ..., r33],
      "cam_t_m2c": [tx, ty, tz],
      "obj_id": 1
    },
    {
      "cam_R_m2c": [...],
      "cam_t_m2c": [...],
      "obj_id": 2
    }
  ],
  "1": [
    ...
  ]
}
```

- Each frame can contain multiple objects
- `cam_R_m2c`: rotation matrix from model to camera (3x3)
- `cam_t_m2c`: translation vector from model to camera (3,)
- `obj_id`: object identifier

---

## ✅ scene_gt_info_camX.json

Stores auxiliary info about each ground truth instance (bounding boxes, visibility, etc.)

```json
{
  "0": [
    {
      "bbox_obj": [x1, y1, x2, y2],
      "bbox_visib": [x1, y1, x2, y2],
      "px_count_all": 4567,
      "px_count_valid": 4500,
      "px_count_visib": 3200,
      "visib_fract": 0.71
    },
    ...
  ],
  "1": [
    ...
  ]
}
```

- `bbox_obj`: tight bounding box around the object
- `bbox_visib`: visible region bounding box
- `px_count_all`: total object pixels in GT mask
- `px_count_valid`: pixels with valid depth
- `px_count_visib`: visible pixels
- `visib_fract`: visible fraction of the object

---

## Number of Files

There are 3 of each file type, one for each camera:

- 3 × `scene_camera_camX.json`
- 3 × `scene_gt_camX.json`
- 3 × `scene_gt_info_camX.json`

Where `X` ∈ [1, 3]
