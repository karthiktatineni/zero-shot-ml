"""
Causal functional mechanisms, random MLPs, linear models, and GP priors for synthetic tabular tasks.
"""

from __future__ import annotations

import numpy as np


def apply_random_activation(z: np.ndarray, act_type: str) -> np.ndarray:
    """Apply non-linear activation function."""
    if act_type == "relu":
        return np.maximum(z, 0.0)
    elif act_type == "tanh":
        return np.tanh(z)
    elif act_type == "gelu":
        # Fast approximation of GELU
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))
    elif act_type == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))
    elif act_type == "leaky_relu":
        return np.where(z > 0, z, 0.1 * z)
    return z


def sample_mlp_task(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate class logits using a random Multi-Layer Perceptron (MLP).
    Architecture: 1-2 layers, random widths (16-48), random activations.
    Returns logits shaped [n_rows, n_classes].
    """
    _n_rows, in_dim = x.shape
    n_layers = rng.integers(1, 3)
    activations = ["relu", "tanh", "gelu", "leaky_relu"]

    current = x
    curr_dim = in_dim

    for _ in range(n_layers):
        out_dim = rng.integers(16, 49)
        std = float(np.sqrt(3.0 / (curr_dim + out_dim)))
        w = rng.normal(0.0, std, size=(curr_dim, out_dim)).astype(np.float32)
        b = rng.normal(0.0, 0.2, size=out_dim).astype(np.float32)
        z = current @ w + b

        act = str(rng.choice(activations))
        current = apply_random_activation(z, act)
        curr_dim = out_dim

    # Output projection to class logits with distinct separation
    w_out = rng.normal(0.0, float(np.sqrt(2.0 / curr_dim)), size=(curr_dim, n_classes)).astype(np.float32)
    b_out = rng.normal(0.0, 0.2, size=n_classes).astype(np.float32)
    logits = current @ w_out + b_out
    return logits.astype(np.float32)


def sample_scm_task(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate target logits using a Structural Causal Model / Directed Acyclic Graph (DAG).
    Features and intermediate latent nodes compute non-linear causal transformations.
    """
    n_rows, in_dim = x.shape
    n_latent_nodes = rng.integers(2, 6)
    total_nodes = in_dim + n_latent_nodes

    # Initialize node matrix [n_rows, total_nodes]
    node_values = np.zeros((n_rows, total_nodes), dtype=np.float32)
    node_values[:, :in_dim] = x

    # Construct DAG relations for latent nodes
    for idx in range(in_dim, total_nodes):
        n_parents = rng.integers(1, min(4, idx + 1))
        parents = rng.choice(idx, size=n_parents, replace=False)

        weights = rng.normal(0.0, 1.2, size=n_parents).astype(np.float32)
        bias = rng.normal(0.0, 0.3)
        noise = rng.normal(0.0, 0.05, size=n_rows).astype(np.float32)

        parent_vals = node_values[:, parents]
        linear_comb = parent_vals @ weights + bias + noise

        op_type = rng.choice(["relu", "tanh", "poly"])
        if op_type == "relu":
            val = np.maximum(linear_comb, 0.0)
        elif op_type == "tanh":
            val = np.tanh(linear_comb)
        else:
            val = np.tanh(linear_comb + 0.1 * (linear_comb ** 2))

        node_values[:, idx] = val

    # Sink projection to logits from latent nodes + subset of root features
    sink_features = np.column_stack([node_values[:, in_dim:], x[:, :min(in_dim, 3)]])
    n_sink_dim = sink_features.shape[1]
    w_sink = rng.normal(0.0, float(np.sqrt(2.0 / n_sink_dim)), size=(n_sink_dim, n_classes)).astype(np.float32)
    b_sink = rng.normal(0.0, 0.2, size=n_classes).astype(np.float32)
    logits = sink_features @ w_sink + b_sink
    return logits.astype(np.float32)


def sample_linear_task(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate target logits using a Generalized Linear Model (GLM) with feature interactions.
    """
    _n_rows, in_dim = x.shape
    sparsity = rng.uniform(0.5, 1.0)
    mask = (rng.uniform(0.0, 1.0, size=(in_dim, n_classes)) < sparsity).astype(np.float32)
    w = rng.normal(0.0, 2.0, size=(in_dim, n_classes)).astype(np.float32) * mask
    b = rng.normal(0.0, 0.5, size=n_classes).astype(np.float32)

    logits = x @ w + b

    # Optional 2nd order pairwise interaction
    if in_dim >= 2 and rng.uniform() < 0.4:
        feat_a, feat_b = rng.choice(in_dim, size=2, replace=False)
        inter = (x[:, feat_a] * x[:, feat_b])[:, None]
        inter_w = rng.normal(0.0, 1.5, size=(1, n_classes)).astype(np.float32)
        logits += inter @ inter_w

    return logits.astype(np.float32)


def sample_gp_task(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate target logits from a Gaussian Process prior via Random Fourier Features (RFF).
    Approximates GP with an RBF / stationary kernel.
    """
    _n_rows, in_dim = x.shape
    n_rff = rng.integers(32, 64)
    lengthscale = rng.uniform(1.5, 4.0)

    # Random spectral frequencies
    w_freq = rng.normal(0.0, 1.0 / lengthscale, size=(in_dim, n_rff)).astype(np.float32)
    bias_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_rff).astype(np.float32)

    # Fourier feature projection: sqrt(2/D) * cos(X W + b)
    phi = np.sqrt(2.0 / n_rff) * np.cos(x @ w_freq + bias_phase)

    # Linear weights sampled from normal
    w_out = rng.normal(0.0, 3.0, size=(n_rff, n_classes)).astype(np.float32)
    b_out = rng.normal(0.0, 0.1, size=n_classes).astype(np.float32)
    logits = phi @ w_out + b_out
    return logits.astype(np.float32)


def sample_tree_task(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate target logits using axis-aligned splits (Decision Tree prior).
    Highly learnable by Random Forests.
    """
    n_rows, in_dim = x.shape
    logits = np.zeros((n_rows, n_classes), dtype=np.float32)
    
    # Use a single feature to guarantee a perfectly axis-aligned split (trivial for Random Forests)
    n_split_feats = 1
    split_feats = rng.choice(in_dim, size=n_split_feats, replace=False)
    
    # Use the selected feature for splitting with tiny jitter to prevent duplicate quantiles
    score = x[:, split_feats[0]] + rng.normal(0.0, 1e-5, size=n_rows)
    # Create C bins
    quantiles = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
    thresholds = np.quantile(score, quantiles)
    labels = np.digitize(score, thresholds)
    
    # Set strong logits for the assigned class
    for c in range(n_classes):
        logits[labels == c, c] = 10.0
        
    return logits


def logits_to_labels(
    logits: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    label_noise: float = 0.0,
) -> np.ndarray:
    """
    Convert raw task logits or continuous latent scores into discrete class labels [0..n_classes-1].
    Supports Argmax decision boundaries, Quantile binning, and Temperature-scaled softmax.
    """
    n_rows = len(logits)
    strategy = rng.choice(["argmax", "quantile", "softmax"], p=[0.75, 0.0, 0.25])

    if strategy == "quantile" or logits.shape[1] == 1:
        # Scalar quantile binning of primary latent direction
        scalar_score = logits[:, 0]
        # Quantile bin boundaries with small jitter

        quantiles = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
        thresholds = np.quantile(scalar_score, quantiles)
        labels = np.digitize(scalar_score, thresholds)
    elif strategy == "softmax":
        # Low temperature softmax sampling for strong learnability
        temp = rng.uniform(0.2, 0.6)
        scaled_logits = logits / temp
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        gumbel_noise = -np.log(-np.log(rng.uniform(1e-5, 1.0 - 1e-5, size=probs.shape)))
        labels = np.argmax(np.log(np.maximum(probs, 1e-12)) + 0.3 * gumbel_noise, axis=-1)
    else:
        # Crisp argmax decision boundary with small boundary noise
        noise = rng.normal(0.0, 0.1, size=logits.shape)
        labels = np.argmax(logits + noise, axis=-1)

    # Optional label noise injection (random label corruption)
    if label_noise > 0.0:
        corrupt_mask = rng.uniform(0.0, 1.0, size=n_rows) < label_noise
        if np.any(corrupt_mask):
            random_classes = rng.integers(0, n_classes, size=int(corrupt_mask.sum()))
            labels[corrupt_mask] = random_classes

    return labels.astype(np.int64)


def sample_task_function(
    x: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
    label_noise: float = 0.0,
) -> np.ndarray:
    """
    Sample synthetic labels for features x using a weighted mixture of priors:
    - 30% Tree (axis-aligned splits, extremely learnable)
    - 30% Linear / GLM
    - 20% MLP
    - 10% SCM / DAG
    - 10% Gaussian Process (RFF)
    """
    # Option C Implementation: Simplify prior mixture for high C tasks to maintain learnability
    if n_classes >= 5:
        prior_type = rng.choice(
            ["tree", "linear", "mlp", "scm"],
            p=[0.40, 0.40, 0.10, 0.10],
        )
    else:
        prior_type = rng.choice(
            ["tree", "linear", "mlp", "scm", "gp"],
            p=[0.30, 0.30, 0.20, 0.10, 0.10],
        )

    if prior_type == "tree":
        logits = sample_tree_task(x, n_classes, rng)
    elif prior_type == "linear":
        logits = sample_linear_task(x, n_classes, rng)
    elif prior_type == "mlp":
        logits = sample_mlp_task(x, n_classes, rng)
    elif prior_type == "scm":
        logits = sample_scm_task(x, n_classes, rng)
    else:
        logits = sample_gp_task(x, n_classes, rng)

    labels = logits_to_labels(logits, n_classes, rng, label_noise=label_noise)
    return labels
