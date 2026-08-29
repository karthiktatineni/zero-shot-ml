import pytest
import json
import os
import tempfile
from pathlib import Path
from zeroshot_pfn.monitoring import TrainingLogger

def test_training_logger_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(tmpdir, filename="test_metrics.jsonl")
        
        # Log a few steps
        logger.log(step=1, loss=1.5, lr=0.001, speed_eps=50.0)
        logger.log(step=2, loss=1.2, lr=0.001, speed_eps=52.0)
        
        filepath = Path(tmpdir) / "test_metrics.jsonl"
        assert filepath.exists()
        
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        record1 = json.loads(lines[0])
        assert record1["step"] == 1
        assert record1["loss"] == 1.5
        assert record1["lr"] == 0.001
        assert record1["speed_eps"] == 50.0
        assert "timestamp" in record1
        
        record2 = json.loads(lines[1])
        assert record2["step"] == 2
        assert record2["loss"] == 1.2
