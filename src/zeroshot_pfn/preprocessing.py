"""
Per-episode robust tabular scaling, categorical encoding, and missingness preprocessing.
"""

from __future__ import annotations

import numpy as np


def robust_scale_features(
    x: np.ndarray,
    missing_mask: np.ndarray | None = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> np.ndarray:
    """
    Robust per-column scaling using quantile clipping and median-IQR normalization.
    Scales and clips active feature values to range [-1.0, 1.0].
    Missing cells (if marked in missing_mask) are filled with 0.0.
    """
    _n_rows, n_cols = x.shape
    scaled_x = np.zeros_like(x, dtype=np.float32)

    for c in range(n_cols):
        col = x[:, c]
        if missing_mask is not None:
            valid_mask = ~missing_mask[:, c]
        else:
            valid_mask = np.isfinite(col)

        valid_vals = col[valid_mask]
        if len(valid_vals) == 0:
            scaled_x[:, c] = 0.0
            continue

        # Quantile clipping to prevent heavy-tail explosion
        val_low = float(np.quantile(valid_vals, q_low))
        val_high = float(np.quantile(valid_vals, q_high))

        if val_high > val_low:
            clipped = np.clip(valid_vals, val_low, val_high)
            med = float(np.median(clipped))
            q75, q25 = np.percentile(clipped, [75, 25])
            iqr = float(q75 - q25)
            scale = iqr if iqr > 1e-6 else float(np.std(clipped) + 1e-6)

            normed = (clipped - med) / (scale * 2.0)
            normed = np.clip(normed, -1.0, 1.0)
            scaled_x[valid_mask, c] = normed.astype(np.float32)
        else:
            # Constant column
            scaled_x[valid_mask, c] = 0.0

        if missing_mask is not None:
            scaled_x[missing_mask[:, c], c] = 0.0

    return scaled_x.astype(np.float32)


def encode_categorical_column(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Ordinal encode a 1D categorical array to integers mapped to [-1, 1].
    """
    unique_vals = np.unique(values)
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded_int = np.array([mapping[v] for v in values], dtype=np.float32)

    n_cats = len(unique_vals)
    if n_cats > 1:
        # Scale integers [0, n_cats-1] to [-1.0, 1.0]
        encoded_scaled = (encoded_int / (n_cats - 1)) * 2.0 - 1.0
    else:
        encoded_scaled = np.zeros_like(encoded_int, dtype=np.float32)

    return encoded_scaled.astype(np.float32), mapping
