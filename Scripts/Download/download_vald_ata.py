import os
import requests
from zipfile import ZipFile

# ---- Config ----
base_url = "https://huggingface.co/datasets/bop-benchmark/ipd/resolve/main"
files = [
    "ipd_base.zip",
    "ipd_models.zip",
    "ipd_val.zip"
]
output_dir = "ipd_data"
extracted_dir = os.path.join(output_dir, "ipd")
os.makedirs(output_dir, exist_ok=True)

# ---- Download Function ----
def download_file(url, dest):
    if os.path.exists(dest):
        print(f"✔ File already exists: {dest}")
        return
    print(f"⬇ Downloading {url}")
    response = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✅ Downloaded: {dest}")

# ---- Extract Function ----
def extract_zip(zip_path, extract_to):
    print(f"📦 Extracting {zip_path} to {extract_to}")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# ---- Main Download Loop ----
for file in files:
    zip_path = os.path.join(output_dir, file)
    file_url = f"{base_url}/{file}"
    download_file(file_url, zip_path)
    extract_zip(zip_path, extracted_dir)

print("✅ All files downloaded and extracted to ./ipd_data/ipd/")
