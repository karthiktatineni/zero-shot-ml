import torch
import torch.nn as nn
import math

class FourierEmbedding(nn.Module):
    """
    Random Fourier Features (RFF) for continuous scalar feature embeddings.
    Maps a scalar x to a high-dimensional vector: [cos(xW), sin(xW)]
    Includes LayerNorm to prevent fp16 overflow in downstream attention layers.
    """
    def __init__(self, d_model: int, sigma: float = 2.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2 for sin/cos pairs"
        self.d_model = d_model
        
        # W is fixed, random frequencies drawn from N(0, sigma^2)
        # Register as buffer so it's moved to device, but not trained (requires_grad=False)
        W = torch.randn(d_model // 2) * sigma
        self.register_buffer("W", W)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D] continuous feature values
        Returns: [B, N, D, d_model] Fourier embeddings
        """
        # x is [B, N, D]. We want to multiply by W [d_model // 2]
        # Resulting shape before sin/cos: [B, N, D, d_model // 2]
        x_proj = x.unsqueeze(-1) * self.W  # Broadcasting
        
        # Apply sin and cos
        out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        
        # Layer norm to keep activations bounded for fp16
        return self.layer_norm(out)

class TabularEmbedding(nn.Module):
    """
    Full feature embedding combining Fourier continuous embeddings 
    and learnable missingness tokens.
    """
    def __init__(self, max_features: int, d_model: int, fourier_scale: float = 2.0):
        super().__init__()
        self.max_features = max_features
        self.d_model = d_model
        
        self.fourier = FourierEmbedding(d_model=d_model, sigma=fourier_scale)
        
        # Learnable vector for each column when the value is missing
        # Shape: [max_features, d_model]
        self.missing_tokens = nn.Parameter(torch.randn(max_features, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, max_features]
        missing_mask: [B, N, max_features] boolean mask (True if missing)
        Returns: [B, N, max_features, d_model]
        """
        B, N, D = x.shape
        assert D == self.max_features, f"Expected {self.max_features} features, got {D}"
        
        # Get Fourier embeddings for the continuous values
        # Shape: [B, N, D, d_model]
        emb = self.fourier(x)
        
        # Expand missing tokens to batch and row dimensions
        # self.missing_tokens is [D, d_model] -> [1, 1, D, d_model]
        missing_expanded = self.missing_tokens.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        
        # Where missing_mask is True, replace the embedding with the missing token
        # missing_mask is [B, N, D] -> [B, N, D, 1]
        missing_mask_expanded = missing_mask.unsqueeze(-1)
        
        out = torch.where(missing_mask_expanded, missing_expanded, emb)
        
        return out
