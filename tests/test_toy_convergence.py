import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from zeroshot_pfn.model.transformer import PFNTransformer

def test_toy_convergence():
    """
    Toy linear convergence test.
    Ensures the model can memorize/learn a simple mapping over 200 steps.
    Verifies that the loss decreases and no NaNs are produced in forward/backward passes.
    """
    B = 2
    N = 30
    D = 5
    max_features = 20
    max_classes = 3
    d_model = 128
    
    torch.manual_seed(42)
    model = PFNTransformer(
        max_features=max_features,
        max_classes=max_classes,
        d_model=d_model,
        n_layers=6,
        n_heads=4,
        d_ff=512
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Generate a fixed, simple synthetic dataset
    # x: [B, N, max_features]
    x = torch.randn(B, N, max_features)
    
    # y depends linearly on the first feature
    # class 0 if x[:, :, 0] < -0.5
    # class 1 if -0.5 <= x[:, :, 0] < 0.5
    # class 2 if x[:, :, 0] >= 0.5
    y = torch.zeros(B, N, dtype=torch.long)
    y[x[:, :, 0] >= -0.5] = 1
    y[x[:, :, 0] >= 0.5] = 2
    
    is_query = torch.zeros(B, N, dtype=torch.bool)
    is_query[:, -10:] = True
    
    missing_mask = torch.zeros(B, N, max_features, dtype=torch.bool)
    missing_mask[:, :, D:] = True
    
    initial_loss = None
    final_loss = None
    
    for step in range(200):
        optimizer.zero_grad()
        
        logits = model(x, y, is_query, missing_mask, num_classes=max_classes)
        
        # Only compute loss on query rows
        # logits: [B, N, max_classes] -> [B*N, max_classes]
        # But we want only where is_query is True
        query_logits = logits[is_query]
        query_y = y[is_query]
        
        loss = criterion(query_logits, query_y)
        
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        
        loss.backward()
        
        # Check for NaN gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient at step {step}"
                
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        if step == 199:
            final_loss = loss.item()
            
    print(f"\nToy Convergence Initial Loss: {initial_loss:.4f}")
    print(f"Toy Convergence Final Loss: {final_loss:.4f}")
            
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    assert final_loss < 0.01, f"Final loss is too high, model failed to converge cleanly: {final_loss:.4f}"

    # Also verify parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    assert 1_000_000 <= total_params <= 4_000_000, f"Parameter count {total_params} is outside acceptable range"
