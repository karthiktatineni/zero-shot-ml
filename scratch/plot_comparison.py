import matplotlib.pyplot as plt
import numpy as np

datasets = ["Iris", "Wine", "Breast Cancer", "Digits", "Titanic"]

# 200k parameters model (d_model=128, 6 layers, 256 d_ff, ~1M params) trained for 200k steps
scores_200k = [0.933, 0.944, 0.936, 0.430, 0.779]

# 10m parameters model (d_model=256, 12 layers, 1107 d_ff, ~10M params) trained for 28k steps
scores_10m = [0.911, 0.963, 0.936, 0.589, 0.382]

# Baseline Models (using best of Random Forest / HistGB from previous benchmark)
scores_baseline = [0.889, 1.000, 0.959, 0.939, 0.814]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, scores_200k, width, label='Local 200k Steps (1M Params)')
rects2 = ax.bar(x, scores_10m, width, label='Modal 28k Steps (10M Params)', color='#2ca02c')
rects3 = ax.bar(x + width, scores_baseline, width, label='Best Baseline (RF/HistGB)', color='#7f7f7f', alpha=0.5)

ax.set_ylabel('Accuracy')
ax.set_title('Zero-Shot PFN Model Scaling Comparison (1M vs 10M Params)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc='lower left')
ax.set_ylim([0, 1.1])

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("v3_10m_comparison.png", dpi=150)
print("Plot saved to v3_10m_comparison.png")
