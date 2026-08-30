import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, r'c:\Users\karth\Desktop\Zero-shot\src')
from zeroshot_pfn.model.transformer import PFNTransformer

def _worker(rank, world_size):
    import tempfile
    store_path = os.path.join(tempfile.gettempdir(), "ddp_store")
    
    # Initialize the process group with FileStore
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    
    # Initialize a small model
    model = PFNTransformer(
        max_features=10,
        max_classes=3,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128
    )
    
    # Wrap in DDP
    ddp_model = DDP(model)
    
    # Create identical inputs
    torch.manual_seed(42)
    B, N, D = 2, 5, 10
    x = torch.randn(B, N, D)
    y = torch.randint(0, 3, (B, N))
    
    is_query = torch.zeros(B, N, dtype=torch.bool)
    is_query[:, 3:] = True # Last 2 rows are query
    
    # Pad missing
    missing_mask = torch.zeros(B, D, dtype=torch.bool)
    missing_mask[:, 8:] = True # Last 2 features missing
    
    # Force padding invisibility testing on the batch level
    # Make batch 1 entirely padded (all query=True, missing=True, to simulate padding)
    is_query[1, :] = True
    missing_mask[1, :] = True
    
    # Forward pass
    logits = ddp_model(x, y, is_query, missing_mask, num_classes=torch.tensor([3, 3]))
    
    # Verify the causal masks actually worked inside the blocks
    # We can check this indirectly: support rows (0..2) should not depend on query rows (3..4)
    # We run forward pass again, altering query rows. Support outputs should remain IDENTICAL.
    x_altered = x.clone()
    x_altered[:, 3:, :] += 100.0 # Change query rows massively
    
    # However, because DDP wraps the model, we want to check if the forward pass yields the exact same
    # intermediate activations or outputs for support rows.
    # We can use a backward pass check.
    
    # Support rows don't produce logits directly, but wait, PFNTransformer returns logits for ALL rows.
    # The masking means logits for row `i` only depend on `0..i-1` if `i` is query.
    # Wait, support rows CAN attend to query rows in the original PFN?
    # NO! Support rows CANNOT attend to query rows. 
    # Let's test this mathematically.
    
    logits1 = ddp_model(x, y, is_query, missing_mask, num_classes=torch.tensor([3, 3]))
    logits2 = ddp_model(x_altered, y, is_query, missing_mask, num_classes=torch.tensor([3, 3]))
    
    # Check if support rows changed
    # (Wait, PFN returns logits for ALL rows, but we only use query rows for loss).
    diff = (logits1[0, :3] - logits2[0, :3]).abs().max()
    
    if diff > 1e-5:
        print(f"[Rank {rank}] FAILED! Support row logits changed when query rows were altered. Diff: {diff}")
        dist.destroy_process_group()
        exit(1)
        
    print(f"[Rank {rank}] PASSED causal masking check under DDP!")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)
