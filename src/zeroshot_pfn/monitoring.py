import json
import time
from pathlib import Path

class TrainingLogger:
    """
    Lightweight, lock-free JSONL logger for training metrics.
    Writes atomic updates by appending single lines.
    This allows a separate dashboard process to read the file safely.
    """
    def __init__(self, run_dir: str | Path, filename: str = "metrics.jsonl"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.run_dir / filename
        
        # Touch file to ensure it exists
        if not self.filepath.exists():
            self.filepath.touch()
            
    def log(self, step: int, loss: float, lr: float, speed_eps: float):
        """
        Appends a metric dictionary as a single JSON line.
        """
        record = {
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "speed_eps": float(speed_eps),
            "timestamp": time.time()
        }
        
        # Append line
        # Using a simple append is generally atomic for small lines (< 4KB) on POSIX and Windows.
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
