import os
import modal

app = modal.App("zeroshot-pfn-10m")

# Create image with specific dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "scikit-learn",
    "pandas"
)

# Set up local workspace mounting using Image.add_local_dir per Modal v1.5 standards
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
image = image.add_local_dir(src_dir, remote_path="/root/src")

# Persistent volume for saving checkpoints
volume = modal.Volume.from_name("zeroshot-pfn-runs", create_if_missing=True)

@app.function(
    image=image, 
    gpu="A100:4", 
    cpu=32.0, 
    timeout=7200,  # 120 minutes for safety
    volumes={"/root/runs": volume}
)
def train_on_modal():
    import subprocess
    
    print("Starting DDP Smoke Test in /root/runs/modal_run_10m_perfect for 500 steps using torchrun...")
    
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "/root/src/zeroshot_pfn/train.py",
        "--run-dir", "/root/runs/modal_run_10m_budget_v6",
        "--total-steps", "28000",
        "--n-layers", "12",
        "--d-model", "256",
        "--n-heads", "8",
        "--d-ff", "1107",
        "--batch-size", "32",
        "--num-workers", "7",
        "--no-compile"  # Disable compile temporarily to isolate NCCL issues
    ]
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    pass
