from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.model.transformer import PFNTransformer

DEFAULT_CHECKPOINT_PATH = Path("runs/main_run/checkpoints/latest.pt")


class ZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn style wrapper for Zero-Shot PFN inference.
    """
    def __init__(self, checkpoint_path: str | None = None, dummy: bool = False, max_n: int = 120, model_kwargs: dict = None):
        self.checkpoint_path = checkpoint_path
        self.dummy = dummy
        self.max_n = max_n
        
        self.config = PriorConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        default_kwargs = {
            "max_features": self.config.n_features_max,
            "max_classes": 10,
            "d_model": 128,
            "n_layers": 6,
            "n_heads": 4,
            "d_ff": 512
        }
        if model_kwargs:
            default_kwargs.update(model_kwargs)
            
        self.model = PFNTransformer(**default_kwargs).to(self.device)
        if self.dummy:
            print("Using DUMMY (untrained) PFN checkpoint for dry run.")
        else:
            checkpoint_path = self._resolve_checkpoint_path(self.checkpoint_path)
            self.checkpoint_path = str(checkpoint_path)
            print(f"Loading PFN checkpoint from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state['model_state_dict'])
            
        self.model.eval()
        self.label_encoder_ = LabelEncoder()

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path:
        """
        Resolve the checkpoint used for inference.

        Running random weights accidentally looks like terrible model accuracy, so a
        missing checkpoint is treated as a configuration error unless dummy=True.
        """
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"PFN checkpoint not found: {checkpoint_path}. "
                "Pass a trained checkpoint path or set dummy=True for an untrained dry run."
            )

        return checkpoint_path
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        PFN doesn't train. It just stores the support set to use as context during predict().
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        pad_cols = self.config.n_features_max - X_t.shape[1]
        if pad_cols > 0:
            X_t = torch.nn.functional.pad(X_t, (0, pad_cols))
            
        y_encoded = self.label_encoder_.fit_transform(y)
            
        self.X_support_ = X_t
        self.y_support_ = torch.tensor(y_encoded, dtype=torch.long)
        self.n_classes_ = len(self.label_encoder_.classes_)
        if self.n_classes_ > 10:
            raise ValueError(f"ZeroShotClassifier only supports up to 10 classes. Got {self.n_classes_} classes.")
            
        return self

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """
        Chunked zero-shot inference.
        """
        X_q = torch.tensor(X_query, dtype=torch.float32)
        pad_cols = self.config.n_features_max - X_q.shape[1]
        if pad_cols > 0:
            X_q = torch.nn.functional.pad(X_q, (0, pad_cols))
            
        preds = []
        
        n_support = self.X_support_.shape[0]
        
        # We can only fit max_n rows in one pass. 
        # Support rows + query rows <= max_n.
        # If n_support is too large, we truncate it via stratified sampling.
        max_support = min(n_support, 80)
        
        if n_support > max_support:
            from sklearn.model_selection import train_test_split
            _, indices = train_test_split(
                np.arange(n_support), 
                train_size=n_support - max_support, 
                test_size=max_support, 
                stratify=self.y_support_.numpy(),
                random_state=42
            )
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = torch.arange(n_support)
            
        x_sup = self.X_support_[indices]
        y_sup = self.y_support_[indices]
        
        chunk_size = self.max_n - max_support
        if chunk_size <= 0:
            raise ValueError(f"Not enough sequence length for query rows. max_n={self.max_n}, max_support={max_support}")
        
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
            for i in range(0, X_q.shape[0], chunk_size):
                x_chunk = X_q[i:i+chunk_size]
                
                # Combine support and query
                # [B, N, D]
                x_batch = torch.cat([x_sup, x_chunk], dim=0).unsqueeze(0).to(self.device)
                
                # y labels for support, 0 for query (will be replaced by query_token_id inside model)
                y_batch = torch.cat([y_sup, torch.zeros(x_chunk.shape[0], dtype=torch.long)], dim=0).unsqueeze(0).to(self.device)
                
                # is_query mask
                iq = torch.cat([torch.zeros(max_support, dtype=torch.bool), torch.ones(x_chunk.shape[0], dtype=torch.bool)], dim=0).unsqueeze(0).to(self.device)
                
                # missing_mask (True for padded columns)
                mm = torch.zeros_like(x_batch, dtype=torch.bool).to(self.device)
                if pad_cols > 0:
                    mm[:, :, -pad_cols:] = True
                
                nc = torch.tensor([self.n_classes_], dtype=torch.long).to(self.device)
                
                # Forward
                logits = self.model(x_batch, y_batch, iq, mm, nc) # [1, N, C]
                
                # Extract query logits
                query_logits = logits[0, max_support:] # [chunk_size, C]
                query_logits = query_logits[:, : self.n_classes_]
                
                chunk_preds = query_logits.argmax(dim=-1).cpu().numpy()
                preds.append(chunk_preds)
                
        preds_concat = np.concatenate(preds)
        return self.label_encoder_.inverse_transform(preds_concat)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "X_support_")
