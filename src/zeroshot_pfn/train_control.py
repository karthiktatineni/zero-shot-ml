import json
import math
import os
from enum import Enum
from pathlib import Path

import torch


class ControlSignal(Enum):
    NONE = 0
    PAUSE = 1
    STOP = 2

def get_control_signal(control_dir: str | Path) -> ControlSignal:
    """Reads the control directory to determine if a PAUSE or STOP signal is active."""
    control_dir = Path(control_dir)
    if not control_dir.exists():
        return ControlSignal.NONE
    
    if (control_dir / "STOP").exists():
        return ControlSignal.STOP
        
    if (control_dir / "PAUSE").exists():
        return ControlSignal.PAUSE
        
    return ControlSignal.NONE

class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path, max_top_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_top_k = max_top_k
        self.top_k_dir = self.checkpoint_dir / "top_k"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep track of top K models (loss -> filepath)
        self.top_k_records = [] 
        
        # Load state if exists
        self.state_file = self.checkpoint_dir / "state.json"
        self.best_loss = float('inf')
        self.latest_val_loss = None
        
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.best_loss = state.get("best_loss", float('inf'))
                self.top_k_records = state.get("top_k_records", [])

    @staticmethod
    def _rankable_loss(val_loss: float | None) -> float | None:
        if val_loss is None:
            return None
        try:
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val_loss):
            return None
        return val_loss

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                "best_loss": self.best_loss,
                "top_k_records": self.top_k_records
            }, f, indent=2)

    def save_checkpoint(
        self, 
        model, 
        optimizer, 
        scaler, 
        scheduler, 
        step: int, 
        val_loss: float,
        is_milestone: bool = False
    ):
        """
        Saves the checkpoint and manages rotation for latest, best, top-k, and milestones.
        """
        rankable_loss = self._rankable_loss(val_loss)

        state_dict = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': rankable_loss
        }
        
        # 1. Always save 'latest.pt'
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state_dict, latest_path)

        # If training is stopped before any loss has been observed, keep latest
        # for resume but do not let an uninitialized metric enter best/top-k.
        if rankable_loss is None:
            self._save_state()
            return
        
        # 2. Check if it's the absolute best
        if rankable_loss < self.best_loss:
            self.best_loss = rankable_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state_dict, best_path)
            
        # 3. Manage top K
        # Check if we should insert into top K
        is_top_k = False
        if len(self.top_k_records) < self.max_top_k or rankable_loss < max([r['loss'] for r in self.top_k_records]):
            is_top_k = True
            
        if is_top_k:
            top_k_path = self.top_k_dir / f"step_{step}_loss_{rankable_loss:.4f}.pt"
            torch.save(state_dict, top_k_path)
            self.top_k_records = [
                r for r in self.top_k_records
                if not (r.get('step') == step and r.get('path') == str(top_k_path))
            ]
            self.top_k_records.append({'step': step, 'loss': rankable_loss, 'path': str(top_k_path)})
            
            # Sort and trim
            self.top_k_records.sort(key=lambda x: x['loss'])
            if len(self.top_k_records) > self.max_top_k:
                removed = self.top_k_records.pop(-1)
                try:
                    os.remove(removed['path'])
                except OSError:
                    pass
                    
        # 4. Save milestone if requested
        if is_milestone:
            milestone_path = self.checkpoint_dir / f"milestone_{step}.pt"
            torch.save(state_dict, milestone_path)
            
        self._save_state()

    def load_latest(self, model, optimizer=None, scaler=None, scheduler=None):
        """Loads the latest checkpoint to resume training."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return 0 # step 0
            
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.latest_val_loss = checkpoint.get('val_loss')
        
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['step']
