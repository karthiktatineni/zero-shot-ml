import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.generator import sample_episode, get_rng

def run_blended_n1000_diagnostic():
    seeds = [2026, 42, 1337]
    
    # Custom config for large N, using native split logic
    config = PriorConfig(
        n_classes_min=10,
        n_classes_max=10,
        n_support_min=1000,
        n_support_max=1000,
        n_query_min=200,
        n_query_max=200,
        n_features_min=20,
        n_features_max=20,
        noise_feature_ratio_min=0.4,
        noise_feature_ratio_max=0.85,
    )
    
    print("Running N=1000 Diagnostic across 3 seeds using native sample_episode logic...")
    
    for seed in seeds:
        rng = get_rng(seed)
        all_f1s = []
        
        for i in range(200):
            # No manual mask overrides - using native episode split logic
            ep = sample_episode(config=config, rng=rng, max_rows=1200)
            
            x_train = ep.x[~ep.is_query][:, ep.feature_mask]
            y_train = ep.y[~ep.is_query]
            x_test = ep.x[ep.is_query][:, ep.feature_mask]
            y_test = ep.y[ep.is_query]
            
            rf = RandomForestClassifier(n_estimators=30, random_state=42)
            rf.fit(x_train, y_train)
            pred = rf.predict(x_test)
            
            f1 = f1_score(y_test, pred, average='macro', zero_division=0)
            all_f1s.append(f1)
            
        print(f"Seed {seed}: N=1000 Blended Mixture C=10 Macro-F1: {np.mean(all_f1s):.4f}")

if __name__ == '__main__':
    run_blended_n1000_diagnostic()
