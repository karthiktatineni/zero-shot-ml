import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.generator import sample_episode
from zeroshot_pfn.model.transformer import PFNTransformer
from zeroshot_pfn.train_control import CheckpointManager, ControlSignal, get_control_signal


class StreamingEpisodeDataset(IterableDataset):
    def __init__(self, config: PriorConfig, max_episodes: int | None = None):
        super().__init__()
        self.config = config
        self.max_episodes = max_episodes

    def __iter__(self):
        count = 0
        while True:
            if self.max_episodes is not None and count >= self.max_episodes:
                break
            # Generate one episode
            ep = sample_episode(self.config)
            
            # Convert to tensors
            x = torch.tensor(ep.x, dtype=torch.float32)
            y = torch.tensor(ep.y, dtype=torch.long)
            is_query = torch.tensor(ep.is_query, dtype=torch.bool)
            
            # Pad features to max_features (20)
            N, D = x.shape
            max_features = self.config.n_features_max
            missing_mask = torch.zeros((N, max_features), dtype=torch.bool)
            
            if D < max_features:
                pad = torch.zeros((N, max_features - D))
                x_padded = torch.cat([x, pad], dim=1)
                missing_mask[:, D:] = True
            else:
                x_padded = x
                
            yield {
                'x': x_padded,
                'y': y,
                'is_query': is_query,
                'missing_mask': missing_mask,
                'n_classes': ep.n_classes
            }
            count += 1

from zeroshot_pfn.monitoring import TrainingLogger


def collate_episodes(batch):
    # Find max N in batch
    max_n = max(item['x'].shape[0] for item in batch)
    
    x_padded = []
    y_padded = []
    is_query_padded = []
    missing_mask_padded = []
    
    for item in batch:
        n = item['x'].shape[0]
        pad_len = max_n - n
        
        # x: pad with 0s
        x_pad = torch.cat([item['x'], torch.zeros((pad_len, item['x'].shape[1]))], dim=0) if pad_len > 0 else item['x']
        x_padded.append(x_pad)
        
        # y: pad with -100
        y_pad = torch.cat([item['y'], torch.full((pad_len,), -100, dtype=torch.long)], dim=0) if pad_len > 0 else item['y']
        y_padded.append(y_pad)
        
        # is_query: pad with True (so they don't affect causal masking of support rows)
        iq_pad = torch.cat([item['is_query'], torch.ones((pad_len,), dtype=torch.bool)], dim=0) if pad_len > 0 else item['is_query']
        is_query_padded.append(iq_pad)
        
        # missing_mask: pad with True
        mm_pad = torch.cat([item['missing_mask'], torch.ones((pad_len, item['missing_mask'].shape[1]), dtype=torch.bool)], dim=0) if pad_len > 0 else item['missing_mask']
        missing_mask_padded.append(mm_pad)
        
    x = torch.stack(x_padded)
    y = torch.stack(y_padded)
    is_query = torch.stack(is_query_padded)
    missing_mask = torch.stack(missing_mask_padded)
    n_classes = torch.tensor([item['n_classes'] for item in batch], dtype=torch.long)
    return x, y, is_query, missing_mask, n_classes

def train_loop(
    run_dir: str = "runs/pilot",
    total_steps: int = 10000,
    batch_size: int = 8,
    log_interval: int = 100,
    accumulation_steps: int = 2
):
    print(f"Starting Training in {run_dir} for {total_steps} steps...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    config = PriorConfig()
    
    # Checkpoint and Control initialization
    run_dir_path = Path(run_dir)
    control_dir = run_dir_path / ".control"
    control_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_mgr = CheckpointManager(run_dir_path / "checkpoints")
    
    # Initialize Logger
    logger = TrainingLogger(run_dir_path)
    
    # Data Loading
    dataset = StreamingEpisodeDataset(config)
    # Using 0 workers for safety in Windows unless if __name__ == "__main__" is heavily respected.
    # We use 0 workers for the pilot to avoid multiprocessing complexities in this demo.
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_episodes,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Model Initialization
    model = PFNTransformer(
        max_features=config.n_features_max,
        max_classes=10,
        d_model=128,
        n_layers=6,
        n_heads=4,
        d_ff=512
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Cosine scheduler with warmup (assuming warmup for first 10% of steps)
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0 # CosineAnnealingLR takes over after warmup (simplified)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Load state if exists
    start_step = checkpoint_mgr.load_latest(model, optimizer, scaler, scheduler)
    moving_loss = checkpoint_mgr.latest_val_loss
    if moving_loss is not None:
        moving_loss = float(moving_loss)
    
    if start_step > 0:
        print(f"Resumed from step {start_step}")
        
    model.train()
    
    data_iter = iter(dataloader)
    
    start_time = time.time()
    
    for step in range(start_step, total_steps):
        # 1. Check Control Signals
        signal = get_control_signal(control_dir)
        if signal == ControlSignal.STOP:
            print("\n[STOP] Signal received. Saving checkpoint and exiting.")
            checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step, moving_loss)
            break
        elif signal == ControlSignal.PAUSE:
            print("\n[PAUSE] Signal received. Saving state and pausing...")
            checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step, moving_loss)
            while get_control_signal(control_dir) == ControlSignal.PAUSE:
                time.sleep(2.0)
            print("[RESUME] Pause signal removed. Resuming training.")
            
        # 2. Get next batch
        x, y, is_query, missing_mask, n_classes = next(data_iter)
        x = x.to(device)
        y = y.to(device)
        is_query = is_query.to(device)
        missing_mask = missing_mask.to(device)
        n_classes = n_classes.to(device)
        
        # 3. Forward and Backward
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x, y, is_query, missing_mask, num_classes=n_classes)
            
            query_logits = logits[is_query]
            query_y = y[is_query]
            
            loss = criterion(query_logits, query_y)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
        # Update metrics
        current_loss = loss.item() * accumulation_steps
        if moving_loss is None:
            moving_loss = current_loss
        else:
            moving_loss = 0.9 * moving_loss + 0.1 * current_loss
        
        # 4. Logging
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            episodes_per_sec = (log_interval * batch_size) / elapsed
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step+1:06d}/{total_steps} | Loss: {moving_loss:.4f} | LR: {lr:.2e} | Speed: {episodes_per_sec:.1f} ep/s")
            
            # Log telemetry to JSONL
            logger.log(step=step+1, loss=moving_loss, lr=lr, speed_eps=episodes_per_sec)
            
            start_time = time.time()
            
        # 5. Checkpointing logic (Milestones)
        is_milestone = (step + 1) in [100_000, 250_000, 500_000]
        # We simulate a "validation" loss for top_k by just using moving_loss for now
        if is_milestone or (step + 1) % 10000 == 0:
            checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step+1, moving_loss, is_milestone)
            
    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a short pilot of 10,000 steps")
    parser.add_argument("--total-steps", type=int, default=100000, help="Total steps to train for (each step is 16 episodes)")
    args = parser.parse_args()
    
    batch_size = 8
    accumulation_steps = 2
    
    if args.pilot:
        steps = 10000
    else:
        steps = args.total_steps
        
    train_loop(
        run_dir="runs/main_run",
        total_steps=steps,
        batch_size=batch_size,
        log_interval=100,
        accumulation_steps=accumulation_steps
    )
