import subprocess
import sys
import os

def run_training(config_path):
    print(f"\n🚀 Running training with config: {config_path}\n")
    subprocess.run([
        sys.executable,
        # "Scripts/Training/E2E_training.py",
        # "Scripts/Training/pose_training_fused.py",
        # "Scripts/Testing/pose_testing_unseen.py",
         "Scripts/Testing/pose_testing_seen.py",
        config_path
    ])

def main():
    # 👇 Hardcoded config paths
    configs = [
        # "Config/config_fusenet_2.yaml",
        "Config/config_fusenet.yaml"
        # "Config/config_fusenet_2_drive.yaml"
    ]

    for config in configs:
        if not os.path.exists(config):
            print(f"❌ Config not found: {config}")
            continue
        run_training(config)

if __name__ == "__main__":
    main()
