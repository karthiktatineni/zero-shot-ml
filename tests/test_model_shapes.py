import torch

from zeroshot_pfn.model.attention import FeatureAttentionBlock, RowAttentionBlock
from zeroshot_pfn.model.transformer import PFNTransformer


def test_pfn_transformer_shapes_and_masking():
    B = 2
    N = 30
    D = 10
    max_features = 20
    max_classes = 10
    d_model = 128
    
    torch.manual_seed(42)
    model = PFNTransformer(
        max_features=max_features,
        max_classes=max_classes,
        d_model=d_model,
        n_layers=2, # 1 pair of feature/row
        n_heads=4,
        d_ff=256
    )
    model.eval()
    
    x = torch.randn(B, N, max_features)
    y = torch.randint(0, max_classes, (B, N))
    is_query = torch.zeros(B, N, dtype=torch.bool)
    # Set last 10 rows as query
    is_query[:, -10:] = True
    
    # Let's say only the first D features are active. The rest are missing/padded.
    missing_mask = torch.zeros(B, N, max_features, dtype=torch.bool)
    missing_mask[:, :, D:] = True
    
    # Active classes
    num_classes = torch.tensor([5, 8]) # Batch 0 has 5 classes, Batch 1 has 8
    
    with torch.no_grad():
        logits1 = model(x, y, is_query, missing_mask, num_classes=num_classes)
    
    # (a) Verify output shape
    assert logits1.shape == (B, N, max_classes), f"Expected {(B, N, max_classes)}, got {logits1.shape}"
    
    # (b) Verify padded/inactive feature columns don't affect outputs
    # Let's perturb the padded feature columns (where missing_mask is True)
    x_perturbed = x.clone()
    x_perturbed[:, :, D:] += 1000.0  # Massive perturbation to inactive features
    
    with torch.no_grad():
        logits2 = model(x_perturbed, y, is_query, missing_mask, num_classes=num_classes)
        
    assert torch.allclose(logits1, logits2, atol=1e-6), \
        "Padded/inactive feature columns affected the output! missing_mask is not working properly."
        
    # (c) Verify inactive class logits are masked to a large negative value.
    # The model uses -10000 instead of -1e9 so mixed-precision inference stays finite.
    assert torch.all(logits1[0, :, 5:] <= -1e4), "Inactive class logits not masked in batch 0"
    assert torch.all(logits1[1, :, 8:] <= -1e4), "Inactive class logits not masked in batch 1"

def test_attention_block_shapes():
    B = 2
    N = 30
    D = 20
    d_model = 128
    
    x = torch.randn(B, N, D, d_model)
    
    feat_block = FeatureAttentionBlock(d_model, n_heads=4, d_ff=256)
    out_feat = feat_block(x)
    assert out_feat.shape == (B, N, D, d_model)
    
    row_block = RowAttentionBlock(d_model, n_heads=4, d_ff=256)
    # create a dummy mask of correct shape [B * D * n_heads, N, N]
    n_heads = 4
    mask = torch.zeros(B * D * n_heads, N, N, dtype=torch.bool)
    out_row = row_block(x, mask=mask)
    assert out_row.shape == (B, N, D, d_model)
