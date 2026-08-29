from unittest.mock import MagicMock

import torch

from zeroshot_pfn.train_control import CheckpointManager, ControlSignal, get_control_signal


def test_get_control_signal(tmp_path):
    control_dir = tmp_path / ".control"
    
    # 1. No directory
    assert get_control_signal(control_dir) == ControlSignal.NONE
    
    control_dir.mkdir()
    
    # 2. Empty directory
    assert get_control_signal(control_dir) == ControlSignal.NONE
    
    # 3. PAUSE file exists
    pause_file = control_dir / "PAUSE"
    pause_file.touch()
    assert get_control_signal(control_dir) == ControlSignal.PAUSE
    
    # 4. STOP file exists (takes precedence if both exist in logic, but let's test separately)
    stop_file = control_dir / "STOP"
    stop_file.touch()
    # If both exist, our current implementation returns STOP
    assert get_control_signal(control_dir) == ControlSignal.STOP
    
    pause_file.unlink()
    assert get_control_signal(control_dir) == ControlSignal.STOP

def test_checkpoint_manager(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir, max_top_k=2)
    
    # Mock model and optimizer
    model = MagicMock()
    model.state_dict.return_value = {"weight": torch.tensor([1.0])}
    
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"step": 0}
    
    scaler = MagicMock()
    scaler.state_dict.return_value = {"scale": 1.0}
    
    scheduler = MagicMock()
    scheduler.state_dict.return_value = {"lr": 0.1}
    
    # Step 1: initial save
    manager.save_checkpoint(model, optimizer, scaler, scheduler, step=10, val_loss=2.5)
    
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "top_k" / "step_10_loss_2.5000.pt").exists()
    
    # Step 2: worse save
    manager.save_checkpoint(model, optimizer, scaler, scheduler, step=20, val_loss=3.0)
    assert len(list((checkpoint_dir / "top_k").glob("*.pt"))) == 2
    
    # Step 3: best save
    manager.save_checkpoint(model, optimizer, scaler, scheduler, step=30, val_loss=1.5, is_milestone=True)
    
    assert (checkpoint_dir / "milestone_30.pt").exists()
    
    # Check top_k eviction
    # It keeps max 2. The losses are 1.5, 2.5, 3.0. So 3.0 should be evicted.
    top_k_files = list((checkpoint_dir / "top_k").glob("*.pt"))
    assert len(top_k_files) == 2
    
    filenames = [f.name for f in top_k_files]
    assert "step_30_loss_1.5000.pt" in filenames
    assert "step_10_loss_2.5000.pt" in filenames
    assert "step_20_loss_3.0000.pt" not in filenames
    
    assert manager.best_loss == 1.5


def test_checkpoint_manager_skips_unrankable_loss(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir, max_top_k=2)

    model = MagicMock()
    model.state_dict.return_value = {"weight": torch.tensor([1.0])}

    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"step": 0}

    manager.save_checkpoint(model, optimizer, scaler=None, scheduler=None, step=1, val_loss=None)

    assert (checkpoint_dir / "latest.pt").exists()
    assert not (checkpoint_dir / "best.pt").exists()
    assert list((checkpoint_dir / "top_k").glob("*.pt")) == []
    assert manager.best_loss == float("inf")
    
def test_load_latest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    
    # Save a fake checkpoint
    state_dict = {
        'step': 42,
        'model_state_dict': {"test": True},
    }
    torch.save(state_dict, checkpoint_dir / "latest.pt")
    
    model = MagicMock()
    step = manager.load_latest(model)
    
    assert step == 42
    assert manager.latest_val_loss is None
    model.load_state_dict.assert_called_with({"test": True})
