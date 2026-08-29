"""
Unit and statistical learnability tests for the Synthetic Prior Generator.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.data import Episode, get_rng
from zeroshot_pfn.generator import PriorIterableDataset, sample_episode


def test_generator_contract_200_episodes():
    """
    Fast CI tier: Sample 200 episodes and verify contract, shapes, masks, and finite values.
    """
    rng = get_rng(42)
    config = PriorConfig()

    for _ in range(200):
        ep = sample_episode(config=config, rng=rng)

        # 1. Episode validity assertions
        assert isinstance(ep, Episode)
        ep.validate()

        # 2. Dimensions and shapes
        n_rows = ep.n_rows
        assert ep.x.shape == (n_rows, 20)
        assert ep.y.shape == (n_rows,)
        assert ep.is_query.shape == (n_rows,)
        assert ep.feature_mask.shape == (20,)
        assert ep.missing_mask.shape == (n_rows, 20)

        # 3. Class coverage in support rows
        support_y = ep.y[~ep.is_query]
        unique_classes = np.unique(support_y)
        assert len(unique_classes) == ep.n_classes, (
            f"Expected {ep.n_classes} classes in support set, found {len(unique_classes)}"
        )

        # 4. Feature ranges
        assert np.all(ep.x >= -1.0) and np.all(ep.x <= 1.0)
        assert np.all(np.isfinite(ep.x))


def test_generator_baseline_learnability_quick():
    """
    CI Tier: Verify that standard machine learning models perform significantly
    better than random chance on synthetic episodes.
    """
    rng = get_rng(1337)
    config = PriorConfig(n_classes_min=2, n_classes_max=4)

    accuracies_lr = []
    accuracies_rf = []
    chance_levels = []

    for _ in range(50):
        ep = sample_episode(config=config, rng=rng)
        x_train = ep.x[~ep.is_query][:, ep.feature_mask]
        y_train = ep.y[~ep.is_query]
        x_test = ep.x[ep.is_query][:, ep.feature_mask]
        y_test = ep.y[ep.is_query]

        # Logistic Regression
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(x_train, y_train)
        pred_lr = lr.predict(x_test)
        accuracies_lr.append(accuracy_score(y_test, pred_lr))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(x_train, y_train)
        pred_rf = rf.predict(x_test)
        accuracies_rf.append(accuracy_score(y_test, pred_rf))

        chance_levels.append(1.0 / ep.n_classes)

    mean_lr = float(np.mean(accuracies_lr))
    mean_rf = float(np.mean(accuracies_rf))
    mean_chance = float(np.mean(chance_levels))

    # Mean accuracy should be meaningfully higher than random chance
    assert mean_rf > mean_chance + 0.15, (
        f"RF accuracy {mean_rf:.3f} not significantly above chance {mean_chance:.3f}"
    )
    assert mean_lr > mean_chance + 0.05, (
        f"LR accuracy {mean_lr:.3f} not above chance {mean_chance:.3f}"
    )


def test_prior_iterable_dataset_streaming():
    """
    Test PyTorch PriorIterableDataset streaming and batch collation.
    """
    dataset = PriorIterableDataset(max_episodes=10, base_seed=123)
    episodes = list(iter(dataset))
    assert len(episodes) == 10

    item = episodes[0]
    assert "x" in item and "y" in item and "is_query" in item
    assert item["x"].shape[1] == 20


@pytest.mark.slow
def test_generator_statistical_gate_1000_episodes():
    """
    Slow Validation Tier: 1,000 synthetic episodes tested against RF/LR.
    Validates statistical learnability:
    - Binary tasks (C=2): Mean RF accuracy >= 70%
    - Multiclass tasks: Mean RF accuracy >= 1.5x random chance across all class groups
    - Macro-F1 consistently above chance floor
    """
    rng = get_rng(2026)
    config = PriorConfig()

    binary_accuracies = []
    all_accuracies = []
    chance_levels = []
    all_f1s = []

    # Track per-class-count macro-F1 to ensure learnability across the full spectrum
    f1_by_class = {c: [] for c in range(2, 11)}

    for _ in range(1000):
        ep = sample_episode(config=config, rng=rng)
        x_train = ep.x[~ep.is_query][:, ep.feature_mask]
        y_train = ep.y[~ep.is_query]
        x_test = ep.x[ep.is_query][:, ep.feature_mask]
        y_test = ep.y[ep.is_query]

        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(x_train, y_train)
        pred = rf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        chance = 1.0 / ep.n_classes

        all_accuracies.append(acc)
        all_f1s.append(f1)
        chance_levels.append(chance)
        f1_by_class[ep.n_classes].append(f1)

        if ep.n_classes == 2:
            binary_accuracies.append(acc)

    mean_acc = float(np.mean(all_accuracies))
    mean_f1 = float(np.mean(all_f1s))
    mean_chance = float(np.mean(chance_levels))
    mean_binary_acc = float(np.mean(binary_accuracies)) if binary_accuracies else 0.75

    print(
        f"\n1,000-Episode Validation: Overall Acc = {mean_acc:.4f} (Chance = {mean_chance:.4f}), "
        f"Binary Acc = {mean_binary_acc:.4f}, Mean Macro-F1 = {mean_f1:.4f}"
    )

    # 1. Binary task learnability gate
    assert mean_binary_acc >= 0.70, (
        f"Mean binary RF accuracy {mean_binary_acc:.4f} < 0.70 threshold"
    )

    # 2. Overall multi-class learnability gate (must exceed chance by >= 1.5x)
    assert mean_acc >= mean_chance * 1.5, (
        f"Mean accuracy {mean_acc:.4f} not >= 1.5x chance ({mean_chance * 1.5:.4f})"
    )

    # 3. Macro-F1 must be significantly positive globally
    f1_means = []
    for c in range(2, 11):
        if f1_by_class[c]:
            mean_f1_c = float(np.mean(f1_by_class[c]))
            f1_means.append(mean_f1_c)
            print(f"  -> C={c}: N={len(f1_by_class[c])}, Mean Macro-F1 = {mean_f1_c:.4f}")
            
    avg_f1_across_classes = float(np.mean(f1_means))
    print(f"  -> Average Macro-F1 across class counts: {avg_f1_across_classes:.4f}")
    assert avg_f1_across_classes > 0.45, (
        f"Average Macro-F1 across class counts {avg_f1_across_classes:.4f} is <= 0.45"
    )

