import argparse
import time
import os
from pathlib import Path

import torch
import torch.amp as amp
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Enable TF32 for large matmul speedups on Ampere/Ada GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.generator import sample_episode
from zeroshot_pfn.model.transformer import PFNTransformer
from zeroshot_pfn.train_control import CheckpointManager, ControlSignal, get_control_signal
from zeroshot_pfn.monitoring import TrainingLogger


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
    run_dir: str,
    total_steps: int,
    batch_size: int,
    num_workers: int,
    log_interval: int,
    accumulation_steps: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    no_compile: bool = False
):
    # Detect if running under torchrun (DDP)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    is_ddp = world_size > 1
    
    if is_ddp:
        os.environ['NCCL_DEBUG'] = 'INFO'
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if rank == 0:
        if is_ddp:
            print(f"Starting DDP Training across {world_size} GPUs in {run_dir} for {total_steps} steps...")
            print(f"Global Batch Size: {batch_size * world_size}")
        else:
            print(f"Starting Single-GPU Training in {run_dir} for {total_steps} steps...")
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    config = PriorConfig()
    
    # Checkpoint and Control initialization (Rank 0 only)
    run_dir_path = Path(run_dir)
    control_dir = run_dir_path / ".control"
    
    checkpoint_mgr = None
    logger = None
    if rank == 0:
        control_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_mgr = CheckpointManager(run_dir_path / "checkpoints")
        logger = TrainingLogger(run_dir_path)
    
    # Data Loading (each rank has its own dataset/dataloader generating unique synthetic data)
    dataset = StreamingEpisodeDataset(config)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_episodes,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    # Model Initialization
    model = PFNTransformer(
        max_features=config.n_features_max,
        max_classes=10,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    
    import sys
    if sys.platform != "win32" and not no_compile:
        if rank == 0:
            print("Enabling torch.compile for maximum throughput...")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    local_cp_mgr = CheckpointManager(run_dir_path / "checkpoints") if rank != 0 else checkpoint_mgr
    start_step = local_cp_mgr.load_latest(model, optimizer, scaler, scheduler)
    moving_loss = local_cp_mgr.latest_val_loss
    if moving_loss is not None:
        moving_loss = float(moving_loss)
    
    if rank == 0 and start_step > 0:
        print(f"Resumed from step {start_step}")
        
    model.train()
    data_iter = iter(dataloader)
    
    start_time = time.time()
    total_data_time = 0.0
    total_gpu_time = 0.0
    
    # Control Signal Tensor [0: RUN, 1: PAUSE, 2: STOP]
    signal_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    
    try:
        for step in range(start_step, total_steps):
            # 1. Check Control Signals (Rank 0 checks and broadcasts)
            if rank == 0:
                signal = get_control_signal(control_dir)
                if signal == ControlSignal.STOP:
                    signal_tensor[0] = 2
                elif signal == ControlSignal.PAUSE:
                    signal_tensor[0] = 1
                else:
                    signal_tensor[0] = 0
            
            if is_ddp:
                dist.broadcast(signal_tensor, src=0)
            
            if signal_tensor[0].item() == 2: # STOP
                if rank == 0:
                    print("\n[STOP] Signal received. Saving checkpoint and exiting.")
                    checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step, moving_loss)
                break
                
            elif signal_tensor[0].item() == 1: # PAUSE
                if rank == 0:
                    print("\n[PAUSE] Signal received. Saving state and pausing...")
                    checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step, moving_loss)
                
                while True:
                    if rank == 0:
                        signal = get_control_signal(control_dir)
                        if signal != ControlSignal.PAUSE:
                            signal_tensor[0] = 0
                        else:
                            signal_tensor[0] = 1
                    
                    if is_ddp:
                        dist.broadcast(signal_tensor, src=0)
                    
                    if signal_tensor[0].item() != 1:
                        if rank == 0:
                            print("[RESUME] Pause signal removed. Resuming training.")
                        break
                    time.sleep(2.0)
                
            # 2. Get next batch
            t0 = time.time()
            x, y, is_query, missing_mask, n_classes = next(data_iter)
            
            x = x.to(device)
            y = y.to(device)
            is_query = is_query.to(device)
            missing_mask = missing_mask.to(device)
            n_classes = n_classes.to(device)
            
            torch.cuda.synchronize(device)
            t1 = time.time()
            total_data_time += (t1 - t0)
            
            # 3. Forward and Backward
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x, y, is_query, missing_mask, num_classes=n_classes)
                query_logits = logits[is_query]
                query_y = y[is_query]
                loss = criterion(query_logits, query_y)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
            torch.cuda.synchronize(device)
            t2 = time.time()
            total_gpu_time += (t2 - t1)
            
            # Update metrics
            if is_ddp:
                current_loss_tensor = torch.tensor(loss.item() * accumulation_steps, device=device)
                dist.all_reduce(current_loss_tensor, op=dist.ReduceOp.AVG)
                current_loss = current_loss_tensor.item()
            else:
                current_loss = loss.item() * accumulation_steps
            
            if moving_loss is None:
                moving_loss = current_loss
            else:
                moving_loss = 0.9 * moving_loss + 0.1 * current_loss
            
            # 4. Logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                episodes_per_sec = (log_interval * batch_size * world_size) / elapsed
                lr = optimizer.param_groups[0]['lr']
                
                total_profiled = total_data_time + total_gpu_time
                data_pct = (total_data_time / total_profiled) * 100 if total_profiled > 0 else 0
                gpu_pct = (total_gpu_time / total_profiled) * 100 if total_profiled > 0 else 0
                
                if rank == 0:
                    print(f"Step {step+1:06d}/{total_steps} | Loss: {moving_loss:.4f} | LR: {lr:.2e} | Speed: {episodes_per_sec:.1f} ep/s | Data: {data_pct:.1f}% | GPU: {gpu_pct:.1f}%")
                    logger.log(step=step+1, loss=moving_loss, lr=lr, speed_eps=episodes_per_sec)
                
                start_time = time.time()
                total_data_time = 0.0
                total_gpu_time = 0.0
                
            # 5. Checkpointing logic (Milestones)
            if rank == 0:
                is_milestone = (step + 1) in [10_000, 50_000, 100_000]
                if is_milestone or (step + 1) % 10000 == 0:
                    checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step+1, moving_loss, is_milestone)
                    
        if rank == 0:
            print("Training finished! Saving final checkpoint...")
            checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, total_steps, moving_loss, True)
            
    except KeyboardInterrupt:
        if rank == 0:
            print("\n[Ctrl+C] Interrupted! Saving latest checkpoint before exiting...")
            if 'step' in locals():
                checkpoint_mgr.save_checkpoint(model, optimizer, scaler, scheduler, step, moving_loss)
    finally:
        if is_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a short pilot of 500 steps")
    parser.add_argument("--total-steps", type=int, default=13750, help="Total steps to train for")
    parser.add_argument("--run-dir", type=str, default="runs/main_run", help="Directory to save the run data")
    
    # 10M Parameter defaults
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1107)
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    
    args = parser.parse_args()
    
    accumulation_steps = 1
    if args.pilot:
        steps = 500
    else:
        steps = args.total_steps
        
    train_loop(
        run_dir=args.run_dir,
        total_steps=steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_interval=100,
        accumulation_steps=accumulation_steps,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        no_compile=args.no_compile
    )
