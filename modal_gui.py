import os
import modal

app = modal.App("zeroshot-pfn-gui")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "streamlit"
)

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
image = image.add_local_dir(src_dir, remote_path="/root/src")

app_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "app.py"))
image = image.add_local_file(app_file, remote_path="/root/app.py")

volume = modal.Volume.from_name("zeroshot-pfn-runs", create_if_missing=True)

@app.function(
    image=image, 
    gpu="any",
    timeout=3600,
    volumes={"/root/runs": volume}
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=100)
def ui():
    import subprocess
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/src"
    
    subprocess.Popen([
        "streamlit", "run", "/root/app.py", 
        "--server.port=8000",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false", 
        "--server.enableXsrfProtection=false"
    ], env=env)
