import subprocess
import sys
import os

def run_training(config_path):
    print(f"\n🚀 Running training with config: {config_path}\n")
    subprocess.run([
        sys.executable,
        "Scripts/train_fusenet.py",
        config_path
    ])

def main():
    # 👇 Hardcoded config paths
    configs = [
        "Config/config_fusenet_2.yaml",
        # "Config/config_fusenet_4.yaml"
    ]

    for config in configs:
        if not os.path.exists(config):
            print(f"❌ Config not found: {config}")
            continue
        run_training(config)

if __name__ == "__main__":
    main()
