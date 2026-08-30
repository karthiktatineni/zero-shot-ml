import os
import modal

app = modal.App("zeroshot-pfn-eval")

# Create image with specific dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "tabulate"
)

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
image = image.add_local_dir(src_dir, remote_path="/root/src")

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
image = image.add_local_dir(data_dir, remote_path="/root/data")

volume = modal.Volume.from_name("zeroshot-pfn-runs", create_if_missing=True)

@app.function(
    image=image, 
    gpu="any", # We don't need A100:4 just for eval, any GPU is fine
    timeout=3600,
    volumes={"/root/runs": volume}
)
def evaluate_on_modal():
    import subprocess
    import os
    
    print("Starting evaluation on Modal...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    
    # 1. Run Evaluation
    cmd_eval = [
        "python", "/root/src/zeroshot_pfn/evaluate.py",
        "--checkpoint", "/root/runs/modal_run_10m_budget_v6/checkpoints/milestone_28000.pt",
        "--out", "/root/runs/modal_run_10m_budget_v6/eval_10m.md",
        "--model-kwargs", "{\"d_model\": 256, \"n_layers\": 12, \"n_heads\": 8, \"d_ff\": 1107}"
    ]
    subprocess.run(cmd_eval, env=env, check=True)
    print("Evaluation completed.")
    
    # Let's also print the evaluation results so we can see them
    with open("/root/runs/modal_run_10m_budget_v6/eval_10m.md", "r") as f:
        print(f.read())
        
    print("All tasks completed on cloud.")

@app.local_entrypoint()
def main():
    evaluate_on_modal.remote()
