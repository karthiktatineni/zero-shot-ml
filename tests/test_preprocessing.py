"""
Unit tests for preprocessing and robust scaling utilities.
"""

from __future__ import annotations

import numpy as np

from zeroshot_pfn.preprocessing import encode_categorical_column, robust_scale_features


def test_robust_scale_features_clipping_and_bounds():
    rng = np.random.default_rng(42)
    # Heavy tailed data with extreme outliers
    data = rng.standard_t(df=2.0, size=(100, 5)).astype(np.float32)
    data[0, 0] = 1000.0
    data[1, 1] = -500.0

    missing_mask = np.zeros((100, 5), dtype=bool)
    missing_mask[5, 2] = True

    scaled = robust_scale_features(data, missing_mask=missing_mask)

    assert scaled.shape == (100, 5)
    assert np.all(scaled >= -1.0) and np.all(scaled <= 1.0)
    assert np.all(np.isfinite(scaled))
    assert scaled[5, 2] == 0.0  # missing filled with 0.0


def test_encode_categorical_column():
    raw_categories = np.array(["cat", "dog", "bird", "dog", "cat", "fish"])
    encoded, mapping = encode_categorical_column(raw_categories)

    assert len(encoded) == 6
    assert len(mapping) == 4
    assert np.all(encoded >= -1.0) and np.all(encoded <= 1.0)
    assert encoded[0] == encoded[4]  # 'cat' has same encoding
    assert encoded[1] == encoded[3]  # 'dog' has same encoding
