import os
import cv2
from tqdm import tqdm
import shutil

base_dir = "ipd_data/ipd/train_pbr"

resize_to = (640, 480)

def should_resize(filename):
    return filename.endswith(".png")

def delete_mask_folders(scene_path):
    for folder in os.listdir(scene_path):
        if folder.startswith("mask_cam") or folder.startswith("mask_visib_cam"):
            full_path = os.path.join(scene_path, folder)
            if os.path.isdir(full_path):
                print(f"Removing {full_path}")
                shutil.rmtree(full_path)

def resize_images(scene_path):
    for folder in os.listdir(scene_path):
        folder_path = os.path.join(scene_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if should_resize(file):
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    resized = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(file_path, resized)

# --- Process all scenes ---
scene_dirs = sorted(os.listdir(base_dir))
for scene in tqdm(scene_dirs, desc="Processing Scenes"):
    scene_path = os.path.join(base_dir, scene)
    if not os.path.isdir(scene_path):
        continue
    delete_mask_folders(scene_path)
    resize_images(scene_path)
