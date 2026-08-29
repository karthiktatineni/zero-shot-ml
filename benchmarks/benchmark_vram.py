import torch
import sys
import gc
from zeroshot_pfn.model.transformer import PFNTransformer

def run_vram_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available. VRAM benchmark cannot be run.")
        sys.exit(0)

    device = torch.device('cuda')
    
    print("======================================================================")
    print("PHASE 3: VRAM STRESS TEST & PERFORMANCE GATE")
    print("======================================================================")
    
    # Max Shape Definition
    B = 8
    N = 120 # N_support = 80, N_query = 40
    D = 20
    C = 10
    
    d_model = 128
    n_layers = 6
    n_heads = 4
    d_ff = 512
    
    print(f"Max Episode Shape: B={B}, N={N}, D={D}, C={C}")
    print(f"Architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Initialize Model
    model = PFNTransformer(
        max_features=D,
        max_classes=C,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params / 1e6:.2f}M")
    
    # 2. Create Dummy Tensors
    x = torch.randn(B, N, D, device=device)
    y = torch.randint(0, C, (B, N), device=device)
    
    is_query = torch.zeros(B, N, dtype=torch.bool, device=device)
    is_query[:, -40:] = True
    
    missing_mask = torch.zeros(B, N, D, dtype=torch.bool, device=device)
    
    # 3. Simulate Forward and Backward Pass with AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x, y, is_query, missing_mask, num_classes=C)
            
            query_logits = logits[is_query]
            query_y = y[is_query]
            
            loss = criterion(query_logits, query_y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("\n--- Backward Pass Completed Successfully ---")
        print(f"Input x shape: {x.shape}")
        print(f"Logits shape before masking: {logits.shape}")
        print(f"Loss value: {loss.item():.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n[FAILED] CUDA Out Of Memory during stress test!")
            print_fallback()
            sys.exit(1)
        else:
            raise e
            
    # 4. Read VRAM
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    
    print(f"Peak VRAM Allocated: {peak_vram_gb:.2f} GB")
    
    if peak_vram_gb > 4.5:
        print("\n[FAILED] VRAM Gate Exceeded! (Limit: 4.5 GB)")
        print_fallback()
        sys.exit(1)
    else:
        print("\n[PASSED] VRAM is comfortably under the 4.5 GB ceiling.")

def print_fallback():
    print("-" * 60)
    print("FALLBACK MATRIX RECOMMENDATIONS:")
    print("To fit within VRAM, systematically apply the following downgrades:")
    print("1. Decrease d_model: 128 -> 96")
    print("2. Decrease d_ff: 512 -> 192")
    print("3. Decrease batch size: 8 -> 4 (and accumulate gradients)")
    print("4. Enable Gradient Checkpointing")
    print("-" * 60)

if __name__ == "__main__":
    run_vram_benchmark()
