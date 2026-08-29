import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.data import get_rng
from zeroshot_pfn.generator import sample_episode

rng = get_rng(2026)
config = PriorConfig()
f1_by_class = {c: [] for c in range(2, 11)}

for i in range(100):
    ep = sample_episode(config=config, rng=rng)
    x_tr = ep.x[~ep.is_query][:, ep.feature_mask]
    y_tr = ep.y[~ep.is_query]
    x_te = ep.x[ep.is_query][:, ep.feature_mask]
    y_te = ep.y[ep.is_query]

    # Test bagging vs RF
    rf = RandomForestClassifier(n_estimators=30, max_features=None, random_state=42)
    rf.fit(x_tr, y_tr)
    pred = rf.predict(x_te)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    f1_by_class[ep.n_classes].append(f1)

for c in range(2, 11):
    if f1_by_class[c]:
        print(f"C={c}: Mean={np.mean(f1_by_class[c]):.4f}")
