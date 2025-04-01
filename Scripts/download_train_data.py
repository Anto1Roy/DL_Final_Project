import os
import subprocess
from huggingface_hub import hf_hub_download

# ---- Config ----
repo_id = "bop-benchmark/ipd"
output_dir = "ipd_data"
extracted_dir = os.path.join(output_dir, "ipd")
os.makedirs(output_dir, exist_ok=True)

# Multi-part zip archive
zip_parts = [
    # "ipd_train_pbr.z01",
    "ipd_train_pbr.z02",
    "ipd_train_pbr.z03",
    "ipd_train_pbr.zip",  # <- this is the last piece
]

# ---- Download Function ----
def download_from_hf(repo_id, filename, dest_dir):
    print(f"⬇ Downloading {filename} from {repo_id}...")
    if( os.path.exists(os.path.join(dest_dir, filename))):
        print(f"✅ Already downloaded: {filename}")
        return os.path.join(dest_dir, filename)
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅ Downloaded: {file_path}")
    return file_path

# ---- Extract Function using 7-Zip ----
def extract_with_7zip(zip_path, output_dir):
    print(f"📦 Extracting {zip_path} using 7-Zip...")
    subprocess.run(["7z", "x", zip_path, f"-o{output_dir}"], check=True)
    print(f"✅ Extracted to {output_dir}")

# ---- Main Logic ----
print("📥 Starting download of all split archive parts...")
for part in zip_parts:
    download_from_hf(repo_id, part, output_dir)

# ---- Extract using 7-Zip ----
final_zip_path = os.path.join(output_dir, "ipd_train_pbr.zip")
extract_with_7zip(final_zip_path, extracted_dir)

print("✅ All files downloaded and extracted to ./ipd_data/ipd/")
