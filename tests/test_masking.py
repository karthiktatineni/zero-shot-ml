import torch
import pytest
from zeroshot_pfn.model.transformer import PFNTransformer

def test_causal_masking_zero_leakage():
    """
    Critical invariance test:
    Changing the features of a query row MUST NOT affect the predictions 
    made for any support row.
    """
    B = 2
    N = 30
    max_features = 20
    max_classes = 10
    d_model = 128
    
    # Must use float32/float64 to avoid fp16 random noise for strict allclose
    torch.manual_seed(42)
    model = PFNTransformer(
        max_features=max_features,
        max_classes=max_classes,
        d_model=d_model,
        n_layers=2, # 1 pair of feature/row
        n_heads=4,
        d_ff=256
    ).float()
    model.eval() # Turn off dropout!
    
    x = torch.randn(B, N, max_features)
    y = torch.randint(0, max_classes, (B, N))
    is_query = torch.zeros(B, N, dtype=torch.bool)
    # Set last 10 rows as query
    is_query[:, -10:] = True
    missing_mask = torch.zeros(B, N, max_features, dtype=torch.bool)
    
    # Baseline forward pass
    with torch.no_grad():
        logits_base = model(x, y, is_query, missing_mask)
        
    # Perturb the query rows dramatically
    x_perturbed = x.clone()
    x_perturbed[is_query] += 1000.0  # Massive perturbation
    
    with torch.no_grad():
        logits_perturbed = model(x_perturbed, y, is_query, missing_mask)
        
    # Check that support row logits are EXACTLY identical
    support_logits_base = logits_base[~is_query]
    support_logits_perturbed = logits_perturbed[~is_query]
    
    # Check invariance
    assert torch.allclose(support_logits_base, support_logits_perturbed, atol=1e-6), \
        "CAUSAL LEAKAGE DETECTED! Support rows were affected by query row changes."

    # Verify query rows DID change
    query_logits_base = logits_base[is_query]
    query_logits_perturbed = logits_perturbed[is_query]
    
    assert not torch.allclose(query_logits_base, query_logits_perturbed, atol=1e-3), \
        "Query rows were unaffected by their own perturbation. Model might be ignoring features."
        
    # --- LIVENESS CHECK: Perturb a support row ---
    # We must ensure that support rows DO attend to each other.
    x_support_perturbed = x.clone()
    
    # Perturb the very first row (which is a support row)
    assert not is_query[0, 0], "Row 0 should be support"
    x_support_perturbed[0, 0] += 1000.0
    
    with torch.no_grad():
        logits_support_perturbed = model(x_support_perturbed, y, is_query, missing_mask)
        
    # Check that OTHER support rows in batch 0 (rows 1 to N-11) HAVE changed
    other_support_logits_base = logits_base[0, 1:20]
    other_support_logits_perturbed = logits_support_perturbed[0, 1:20]
    
    assert not torch.allclose(other_support_logits_base, other_support_logits_perturbed, atol=1e-3), \
        "SUPPORT->SUPPORT MASKING ERROR! Other support rows were not affected by a support row perturbation. Mask is too restrictive."

def test_causal_masking_padding_invisibility():
    """
    Critical invariance test for padding:
    Changing the features of a PADDED row MUST NOT affect the predictions 
    made for ANY valid row (Support or Query).
    Padded rows must be completely invisible to the attention mechanism.
    """
    B = 2
    N = 60 # max row size
    max_features = 20
    max_classes = 10
    d_model = 128
    
    torch.manual_seed(42)
    model = PFNTransformer(
        max_features=max_features,
        max_classes=max_classes,
        d_model=d_model,
        n_layers=2, 
        n_heads=4,
        d_ff=256
    ).float()
    model.eval() 
    
    x = torch.randn(B, N, max_features)
    y = torch.randint(0, max_classes, (B, N))
    is_query = torch.zeros(B, N, dtype=torch.bool)
    missing_mask = torch.zeros(B, N, max_features, dtype=torch.bool)
    
    # Episode 0: all N=60 are valid (e.g. 40 support, 20 query)
    is_query[0, 40:] = True
    
    # Episode 1: only first 30 are valid (20 support, 10 query), last 30 are padding
    is_query[1, 20:30] = True
    # Mark the last 30 as padding using the exact conventions from collate_episodes
    # y = -100, x = 0, is_query = True, missing_mask = True
    y[1, 30:] = -100
    x[1, 30:] = 0.0
    is_query[1, 30:] = True
    missing_mask[1, 30:] = True
    
    # Baseline forward pass
    with torch.no_grad():
        logits_base = model(x, y, is_query, missing_mask)
        
    # Perturb the padded rows dynamically
    x_perturbed = x.clone()
    x_perturbed[1, 30:] += 1000.0  # Massive perturbation to padding
    
    with torch.no_grad():
        logits_perturbed = model(x_perturbed, y, is_query, missing_mask)
        
    # Valid rows in Episode 1 are indices 0 to 29 (Support and Query)
    valid_logits_base = logits_base[1, :30]
    valid_logits_perturbed = logits_perturbed[1, :30]
    
    # Assert padding perturbation did not leak into valid rows
    assert torch.allclose(valid_logits_base, valid_logits_perturbed, atol=1e-6), \
        "PADDING LEAKAGE DETECTED! Valid rows were affected by padding row changes."
    
    # Verify that if we perturb a VALID row, it DOES affect valid queries (liveness)
    x_liveness = x.clone()
    x_liveness[1, 0] += 1000.0 # Perturb first support row
    
    with torch.no_grad():
        logits_liveness = model(x_liveness, y, is_query, missing_mask)
        
    valid_query_logits_base = logits_base[1, 20:30]
    valid_query_logits_liveness = logits_liveness[1, 20:30]
    
    assert not torch.allclose(valid_query_logits_base, valid_query_logits_liveness, atol=1e-3), \
        "Padding mask liveness failed. Valid queries were not affected by valid support."
