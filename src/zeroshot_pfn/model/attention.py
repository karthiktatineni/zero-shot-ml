import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Standard Transformer MLP using GELU activation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class FeatureAttentionBlock(nn.Module):
    """
    Self-attention across the feature (D) dimension.
    Input: [B, N, D, d_model]
    Treats B*N as the effective batch size, allowing features to interact 
    within each row independently.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, N, D, d_model]
        B, N, D, d = x.shape
        
        # Reshape to [B*N, D, d_model]
        x_flat = x.reshape(B * N, D, d)
        # Pre-LN
        norm_x = self.norm1(x_flat)
        
        # QKV Projection
        # [B*N, D, 3 * d_model]
        qkv = self.qkv(norm_x)
        # Reshape to [B*N, D, 3, n_heads, head_dim]
        qkv = qkv.reshape(B * N, D, 3, self.n_heads, self.d_model // self.n_heads)
        # Permute to [3, B*N, n_heads, D, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # FlashAttention / Memory-Efficient Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        
        # Restore shape: [B*N, D, d_model]
        attn_out = attn_out.transpose(1, 2).reshape(B * N, D, self.d_model)
        attn_out = self.proj(attn_out)
        
        # Residual
        x_flat = x_flat + attn_out
        
        # MLP Pre-LN
        norm_x = self.norm2(x_flat)
        mlp_out = self.mlp(norm_x)
        
        # Residual
        x_flat = x_flat + mlp_out
        
        # Restore shape
        return x_flat.reshape(B, N, D, d)

class RowAttentionBlock(nn.Module):
    """
    Self-attention across the row (N) dimension.
    Input: [B, N, D, d_model]
    Treats B*D as the effective batch size, allowing rows to interact
    within each feature dimension independently.
    Applies causal masking to prevent query rows from leaking labels.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        mask: [N, N] boolean mask. True means 'do not attend'.
              Specifically, mask[i, j] = True means row i CANNOT attend to row j.
        """
        # x is [B, N, D, d_model]
        B, N, D, d = x.shape
        
        # Permute and reshape to [B*D, N, d_model]
        # We want sequence length to be N
        x_flat = x.transpose(1, 2).reshape(B * D, N, d)
        # Pre-LN
        norm_x = self.norm1(x_flat)
        
        # QKV Projection
        # [B*D, N, 3 * d_model]
        qkv = self.qkv(norm_x)
        qkv = qkv.reshape(B * D, N, 3, self.n_heads, self.d_model // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B*D, n_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if mask is not None:
            # Our custom mask: True means DO NOT ATTEND
            # SDPA boolean mask: True means ATTEND
            # mask is [B*D*n_heads, N, N]
            attn_mask = ~mask
            # Reshape to [B*D, n_heads, N, N]
            attn_mask = attn_mask.reshape(B * D, self.n_heads, N, N)
        else:
            attn_mask = None

        # FlashAttention / Memory-Efficient Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout_p
        )
        
        # Restore shape: [B*D, N, d_model]
        attn_out = attn_out.transpose(1, 2).reshape(B * D, N, self.d_model)
        attn_out = self.proj(attn_out)
        
        # Residual
        x_flat = x_flat + attn_out
        
        # MLP Pre-LN
        norm_x = self.norm2(x_flat)
        mlp_out = self.mlp(norm_x)
        
        # Residual
        x_flat = x_flat + mlp_out
        
        # Restore shape [B, D, N, d_model] -> [B, N, D, d_model]
        return x_flat.reshape(B, D, N, d).transpose(1, 2)
