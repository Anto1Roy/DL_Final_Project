# Format of BOP datasets

This file describes the [BOP-scenewise](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_scenewise.py) dataset format. This format can be converted to the [BOP-imagewise](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_imagewise.py) format using script [convert_scenewise_to_imagewise.py](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/convert_scenewise_to_imagewise.py) and to the [BOP-webdataset](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/bop_webdataset.py) format using script [convert_imagewise_to_webdataset.py](https://github.com/thodan/bop_toolkit/tree/master/bop_toolkit_lib/dataset/convert_imagewise_to_webdataset.py).

Datasets provided on the [BOP website](https://bop.felk.cvut.cz/datasets) are in the BOP-scenewise format with exception of the [MegaPose training datasets](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_challenge_2023_training_datasets.md) provided for BOP Challenge 2023, which are in the BOP-webdataset format.

## Directory structure

The datasets have the following structure:

```
DATASET_NAME
├─ camera[_CAMTYPE].json
├─ dataset_info.json
├─ test_targets_bop19.json
├─ test_targets_bop24.json
├─ [test_targets_multiview_bop25.json]
├─ models[_MODELTYPE][_eval]
│  ├─ models_info.json
│  ├─ obj_OBJ_ID.ply
├─ train|val|test[_SPLITTYPE]|onboarding_static|onboarding_dynamic
│  ├─ SCENE_ID|OBJ_ID
│  │  ├─ scene_camera[_CAMTYPE].json
│  │  ├─ scene_gt[_CAMTYPE]son
│  │  ├─ scene_gt_info[_CAMTYPE].json
│  │  ├─ scene_gt_coco[_CAMTYPE].json
│  │  ├─ depth[_CAMTYPE]
│  │  ├─ mask[_CAMTYPE]
│  │  ├─ mask_visib[_CAMTYPE]
│  │  ├─ rgb|gray[_CAMTYPE]
```

[_SPLITTYPE] and [_CAMTYPE] are defined to be sensor and/or modality names in multi-sensory datasets.

- _models[\_MODELTYPE]_ - 3D object models.
- _models[\_MODELTYPE]\_eval_ - "Uniformly" resampled and decimated 3D object
  models used for calculation of errors of object pose estimates.

- _train[\_TRAINTYPE]/X_ (optional) - Training images of object X.
- _val[\_VALTYPE]/Y_ (optional) - Validation images of scene Y.
- _test[\_TESTTYPE]/Y_ - Test images of scene Y.
- _onboarding_static/obj_X_SIDE_ - Only for model-free tasks, static onboarding images of object X at up/down side.
- _onboarding_dynamic/obj_X_ - Only for model-free tasks, dynamic onboarding images of object X.

- _camera.json_ - Camera parameters (for sensor simulation only; per-image
  camera parameters are in files _scene_camera.json_ - see below).
- _dataset_info.md_ - Dataset-specific information.
- _test_targets_bop19.json_ - A list of test targets used for the localization evaluation since the BOP Challenge 2019.
- _test_targets_bop24.json_ - A list of test targets used for the detection evaluation since the BOP Challenge 2024.
- _test_targets_multiview_bop25.json_ - A list of test targets used for the multi-view detection evaluation since the BOP Challenge 2025.

_MODELTYPE_, _TRAINTYPE_, _VALTYPE_ and _TESTTYPE_ are optional and used if more
data types are available (e.g. images from different sensors).

The images in _train_, _val_ and _test_ folders are organized into subfolders:

- _rgb/gray_ - Color/gray images.
- _depth_ - Depth images (saved as 16-bit unsigned short).
- _mask_ (optional) - Masks of object silhouettes.
- _mask_visib_ (optional) - Masks of the visible parts of object silhouettes.

The corresponding images across the subolders have the same ID, e.g.
_rgb/000000.png_ and _depth/000000.png_ is the color and the depth image
of the same RGB-D frame. The naming convention for the masks is IMID_GTID.png,
where IMID is an image ID and GTID is the index of the ground-truth annotation
(stored in _scene_gt.json_).

## Training, validation and test images

If both validation and test images are available for a dataset, the ground-truth
annotations are public only for the validation images. Performance scores for
test images with private ground-truth annotations can be calculated in the
[BOP evaluation system](http://bop.felk.cvut.cz).

### Camera parameters

Each set of images is accompanied with file _scene_camera.json_ which contains
the following information for each image:

- _cam_K_ - 3x3 intrinsic camera matrix K (saved row-wise).
- _depth_scale_ - Multiply the depth image with this factor to get depth in mm.
- _cam_R_w2c_ (optional) - 3x3 rotation matrix R_w2c (saved row-wise).
- _cam_t_w2c_ (optional) - 3x1 translation vector t_w2c.
- _view_level_ (optional) - Viewpoint subdivision level, see below.

The matrix K may be different for each image. For example, the principal point
is not constant for images in T-LESS as the images were obtained by cropping a
region around the projection of the origin of the world coordinate system.

Note that the intrinsic camera parameters can be found also in file
_camera.json_ in the root folder of a dataset. These parameters are meant only
for simulation of the used sensor when rendering training images.

P_w2i = K _ [R\_w2c, t\_w2c] is the camera matrix which transforms 3D point
p_w = [x, y, z, 1]' in the world coordinate system to 2D point p_i =
[u, v, 1]' in the image coordinate system: s _ p_i = P_w2i \* p_w.

### Ground-truth annotations

The ground truth object poses are provided in files _scene_gt.json_ which
contain the following information for each annotated object instance:

- _obj_id_ - Object ID.
- _cam_R_m2c_ - 3x3 rotation matrix R_m2c (saved row-wise).
- _cam_t_m2c_ - 3x1 translation vector t_m2c.

P_m2i = K _ [R\_m2c, t\_m2c] is the camera matrix which transforms 3D point
p_m = [x, y, z, 1]' in the model coordinate system to 2D point p_i =
[u, v, 1]' in the image coordinate system: s _ p_i = P_m2i \* p_m.

Ground truth bounding boxes and instance masks are also provided in COCO format under _scene_gt_coco.json_. The RLE format is used for segmentations. Detailed information about the COCO format can be found [here](https://cocodataset.org/#format-data).

### Meta information about the ground-truth poses

The following meta information about the ground-truth poses is provided in files
_scene_gt_info.json_ (calculated using _scripts/calc_gt_info.py_, with delta =
5mm for ITODD, 15mm for other datasets, and 5mm for all photorealistic training
images provided for the BOP Challenge 2020):

- _bbox_obj_ - 2D bounding box of the object silhouette given by (x, y, width,
  height), where (x, y) is the top-left corner of the bounding box.
- _bbox_visib_ - 2D bounding box of the visible part of the object silhouette.
- _px_count_all_ - Number of pixels in the object silhouette.
- _px_count_valid_ - Number of pixels in the object silhouette with a valid
  depth measurement (i.e. with a non-zero value in the depth image).
- _px_count_visib_ - Number of pixels in the visible part of the object
  silhouette.
- _visib_fract_ - The visible fraction of the object silhouette (= _px_count_visib_/_px_count
  \_all_).

## Acquisition of training images

Most of the datasets include training images which were obtained either by
capturing real objects from various viewpoints or by rendering 3D object models
(using _scripts/render_train_imgs.py_).

The viewpoints, from which the objects were rendered, were sampled from a view
sphere as in [2] by recursively subdividing an icosahedron. The level of
subdivision at which a viewpoint was added is saved in _scene_camera.json_ as
_view_level_ (viewpoints corresponding to vertices of the icosahedron have
_view_level_ = 0, viewpoints obtained in the first subdivision step have
_view_level_ = 1, etc.). To reduce the number of viewpoints while preserving
their "uniform" distribution over the sphere surface, one can consider only
viewpoints with _view_level_ <= n, where n is the highest considered level of
subdivision.

For rendering, the radius of the view sphere was set to the distance of the
closest occurrence of any annotated object instance over all test images. The
distance was calculated from the camera center to the origin of the model
coordinate system.

## 3D object models

The 3D object models are provided in PLY (ascii) format. All models include
vertex normals. Most of the models include also vertex color or vertex texture
coordinates with the texture saved as a separate image.
The vertex normals were calculated using
[MeshLab](http://meshlab.sourceforge.net/) as the angle-weighted sum of face
normals incident to a vertex [4].

Each folder with object models contains file _models_info.json_, which includes
the 3D bounding box and the diameter for each object model. The diameter is
calculated as the largest distance between any pair of model vertices.

## Coordinate systems

All coordinate systems (model, camera, world) are right-handed.
In the model coordinate system, the Z axis points up (when the object is
standing "naturally up-right") and the origin coincides with the center of the
3D bounding box of the object model.
The camera coordinate system is as in
[OpenCV](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
with the camera looking along the Z axis.

## Units

- Depth images: See files _camera.json/scene_camera.json_ in individual
  datasets.
- 3D object models: 1 mm
- Translation vectors: 1 mm

## References

[1] Hodan, Michel et al. "BOP: Benchmark for 6D Object Pose Estimation" ECCV'18.

[2] Hinterstoisser et al. "Model based training, detection and pose estimation
of texture-less 3d objects in heavily cluttered scenes" ACCV'12.

[3] Thurrner and Wuthrich "Computing vertex normals from polygonal
facets" Journal of Graphics Tools 3.1 (1998).
