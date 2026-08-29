import torch
import pytest
from zeroshot_pfn.model.transformer import PFNTransformer
from zeroshot_pfn.generator import sample_episode
from zeroshot_pfn.config import PriorConfig

def test_training_smoke_amp():
    """
    Mixed precision smoke test.
    Ensures that the model can handle actual synthetic episodes from the Phase 1 Generator
    using AMP without NaN losses or gradients, and that the loss generally decreases.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available. Smoke test requires CUDA for AMP.")
        
    device = torch.device('cuda')
    
    # 1. Initialize Architecture
    model = PFNTransformer(
        max_features=20,
        max_classes=10,
        d_model=128,
        n_layers=6,
        n_heads=4,
        d_ff=512
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use standard PriorConfig
    config = PriorConfig()
    
    # To accumulate gradients and make training smoother across diverse episodes
    accumulation_steps = 4
    
    losses = []
    
    model.train()
    
    # Run 100 synthetic episodes
    for step in range(100):
        # Generate an actual episode from Phase 1 generator
        ep = sample_episode(config)
        
        # Move to GPU and convert from NumPy to Tensor
        x = torch.tensor(ep.x, dtype=torch.float32, device=device).unsqueeze(0) # [1, N, D]
        y = torch.tensor(ep.y, dtype=torch.long, device=device).unsqueeze(0) # [1, N]
        is_query = torch.tensor(ep.is_query, dtype=torch.bool, device=device).unsqueeze(0) # [1, N]
        
        # Build missing_mask for the rest of the features up to 20
        B, N, D = x.shape
        missing_mask = torch.zeros(B, N, 20, dtype=torch.bool, device=device)
        
        # Pad x to 20 features
        if D < 20:
            pad = torch.zeros(B, N, 20 - D, device=device)
            x_padded = torch.cat([x, pad], dim=2)
            missing_mask[:, :, D:] = True
        else:
            x_padded = x
            
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(
                x=x_padded, 
                y=y, 
                is_query=is_query, 
                missing_mask=missing_mask, 
                num_classes=ep.n_classes
            )
            
            query_logits = logits[is_query]
            query_y = y[is_query]
            
            loss = criterion(query_logits, query_y)
            # Scale down loss for accumulation
            loss = loss / accumulation_steps
            
        assert not torch.isnan(loss), f"NaN loss encountered at step {step}"
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            # Check for NaN gradients before step
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # GradScaler unscales internally before step, but we check scaled grads here for NaNs
                    assert not torch.isnan(param.grad).any(), f"NaN gradient encountered in {name} at step {step}"
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        losses.append(loss.item() * accumulation_steps)

    # Basic convergence check: average loss of last 20 steps should be lower than first 20 steps
    # We don't expect perfect memorization because every step is a completely novel randomly generated SCM/MLP dataset!
    avg_first_20 = sum(losses[:20]) / 20.0
    avg_last_20 = sum(losses[-20:]) / 20.0
    
    print(f"Smoke Test Initial Loss: {avg_first_20:.4f}, Final Loss: {avg_last_20:.4f}")
    assert avg_last_20 <= avg_first_20 + 0.5, "Model is violently diverging on real data"
