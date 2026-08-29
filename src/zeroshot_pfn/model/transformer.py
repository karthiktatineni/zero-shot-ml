import torch
import torch.nn as nn
from .embeddings import TabularEmbedding
from .attention import FeatureAttentionBlock, RowAttentionBlock

def create_causal_mask(is_query: torch.Tensor, is_padding: torch.Tensor, num_features: int, num_heads: int) -> torch.Tensor:
    """
    Creates a causal boolean mask for Row Attention to prevent label leakage.
    Rules:
    - Support rows attend ONLY to Support rows.
    - Query rows attend to ALL rows (Support + Query).
    - Forbidden: Support rows attending to Query rows.
    - Forbidden: ANY row attending to Padding rows.
    
    is_query: [B, N] boolean tensor. True if the row is a query row.
    is_padding: [B, N] boolean tensor. True if the row is a padded row.
    Returns: [B * D * num_heads, N, N] boolean mask (True = do not attend).
    """
    B, N = is_query.shape
    D = num_features
    
    # mask[b, i, j] is True if row i is Support and row j is Query
    # Support: ~is_query, Query: is_query
    # shape: [B, N, 1] & [B, 1, N] -> [B, N, N]
    is_support = ~is_query & ~is_padding
    mask = is_support.unsqueeze(2) & is_query.unsqueeze(1)
    
    # Prevent anyone from attending to padding rows
    mask = mask | is_padding.unsqueeze(1)
    
    # MultiheadAttention with batch_first=True expects 3D mask of shape 
    # [batch_size * num_heads, target_len, source_len]
    # Here, the 'batch_size' for RowAttention is B * D.
    # So we need shape [B * D * num_heads, N, N].
    
    # Expand to [B, D, num_heads, N, N]
    mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand(B, D, num_heads, N, N)
    
    # Reshape to [B * D * num_heads, N, N]
    return mask_expanded.reshape(B * D * num_heads, N, N)


class PFNTransformer(nn.Module):
    """
    Zero-Shot Tabular Prior-Fitted Network (PFN).
    Processes small tabular datasets in a single forward pass.
    """
    def __init__(
        self, 
        max_features: int = 20, 
        max_classes: int = 10, 
        d_model: int = 128, 
        n_layers: int = 6, 
        n_heads: int = 4, 
        d_ff: int = 512, 
        dropout: float = 0.0,
        fourier_scale: float = 2.0
    ):
        super().__init__()
        self.max_features = max_features
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Feature embeddings (continuous values + missingness)
        self.tab_emb = TabularEmbedding(
            max_features=max_features, 
            d_model=d_model, 
            fourier_scale=fourier_scale
        )
        
        # Label embedding (Classes 0 to max_classes-1, plus 1 for the Query masked token)
        self.query_token_id = max_classes
        self.label_emb = nn.Embedding(max_classes + 1, d_model)
        
        # Alternating Dual-Axis Attention Blocks
        # n_layers usually means pairs of (Feature, Row) blocks, or total blocks.
        # Plan says: "6 total attention blocks (3 feature, 3 row)"
        assert n_layers % 2 == 0, "n_layers must be even for alternating blocks"
        n_pairs = n_layers // 2
        
        self.blocks = nn.ModuleList()
        for _ in range(n_pairs):
            self.blocks.append(FeatureAttentionBlock(d_model, n_heads, d_ff, dropout))
            self.blocks.append(RowAttentionBlock(d_model, n_heads, d_ff, dropout))
            
        # Output Head
        # Predict logits for max_classes
        self.head = nn.Linear(d_model, max_classes)
        
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        is_query: torch.Tensor, 
        missing_mask: torch.Tensor,
        num_classes: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        """
        x: [B, N, max_features] continuous feature values
        y: [B, N] categorical labels (0 to max_classes-1). 
           For query rows, these values are ignored and replaced with query_token_id.
        is_query: [B, N] boolean mask indicating query rows
        missing_mask: [B, N, max_features] boolean mask indicating missing features
        num_classes: [B] or int. Number of active classes in this episode. 
                     Logits for classes >= num_classes will be masked to -1e9.
        
        Returns:
            logits: [B, N, max_classes] predictions for all rows
        """
        B, N, D = x.shape
        
        # 1. Feature Embeddings
        # [B, N, D, d_model]
        h = self.tab_emb(x, missing_mask)
        
        # Determine padding mask BEFORE y is overwritten
        is_padding = (y == -100)
        
        # 2. Label Embeddings
        # Mask out labels for query rows
        y_masked = y.clone()
        y_masked[is_query] = self.query_token_id
        
        # [B, N, d_model] -> [B, N, 1, d_model]
        y_emb = self.label_emb(y_masked).unsqueeze(2)
        
        # Add label embeddings to all feature columns for each row
        h = h + y_emb
        
        # 3. Transformer Blocks
        causal_mask = create_causal_mask(is_query, is_padding, D, self.n_heads).to(x.device)
        
        for idx, block in enumerate(self.blocks):
            if isinstance(block, FeatureAttentionBlock):
                h = block(h)
            elif isinstance(block, RowAttentionBlock):
                h = block(h, mask=causal_mask)
                
        # 4. Output Head
        # We need to pool across features to make a row-level prediction.
        # Average pooling across the feature dimension (D)
        # [B, N, D, d_model] -> [B, N, d_model]
        h_pooled = h.mean(dim=2)
        
        # [B, N, max_classes]
        logits = self.head(h_pooled)
        
        # 5. Mask Inactive Classes
        if num_classes is not None:
            if isinstance(num_classes, int):
                # Mask out any class >= num_classes
                if num_classes < self.head.out_features:
                    logits[:, :, num_classes:] = -10000.0
            else:
                # num_classes is a tensor [B]
                # Create a mask of valid classes [B, max_classes]
                arange = torch.arange(self.head.out_features, device=x.device).unsqueeze(0)
                mask = arange >= num_classes.unsqueeze(1)
                # mask is True for inactive classes. Shape [B, max_classes]
                # Expand to [B, N, max_classes]
                mask = mask.unsqueeze(1).expand(-1, N, -1)
                logits = logits.masked_fill(mask, -10000.0)
        
        return logits
