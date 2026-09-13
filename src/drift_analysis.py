"""
Concept drift: per-timestep F1, not just one aggregate test-period number.

A single train/test split (train <=34, test >34) reports one F1 for the
whole test period, which hides whether performance is stable over time or
concentrated in a few good/bad timesteps. Elliptic is documented in the
literature as having a real distribution shift around timestep 43 (widely
attributed to a dark web marketplace shutdown changing the mix of
transaction patterns) - this script checks whether that shock actually
shows up in this project's own model, rather than assuming it does.

Uses the already-computed baseline predictions from data/predictions.pkl
(from train_model.py) - no retraining needed, this is purely an analysis
of existing predictions grouped by timestep.
"""
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score

from results_io import set_drift_analysis

features = pd.read_csv('data/elliptic_txs_features.csv', header=None, usecols=[0, 1])
features.columns = ['txId', 'timestep']
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
df = features.merge(classes, on='txId', how='left')
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

with open('data/predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

df['baseline_proba'] = df['txId'].map(lambda x: predictions.get(x, {}).get('baseline'))
df = df.dropna(subset=['baseline_proba'])
df['baseline_pred'] = (df['baseline_proba'] >= 0.5).astype(int)

print("Per-timestep metrics for the baseline model (threshold 0.5):\n")
per_timestep = []
for t in sorted(df['timestep'].unique()):
    sub = df[df['timestep'] == t]
    n_illicit = int((sub['class'] == 1).sum())
    if n_illicit == 0:
        # Precision/recall/F1 for the illicit class are undefined with no
        # positive examples in this timestep - skip rather than report a
        # misleading 0.0 or 1.0.
        continue
    p = precision_score(sub['class'], sub['baseline_pred'], pos_label=1, zero_division=0)
    r = recall_score(sub['class'], sub['baseline_pred'], pos_label=1, zero_division=0)
    f1 = f1_score(sub['class'], sub['baseline_pred'], pos_label=1, zero_division=0)
    per_timestep.append({
        "timestep": int(t), "n_labeled": int(len(sub)), "n_illicit": n_illicit,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "split": "train" if t <= 34 else "test",
    })
    print(f"  t={t:2d} ({'train' if t <= 34 else 'test '}) n={len(sub):5d} illicit={n_illicit:4d} "
          f"precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

# Flag the worst F1 timesteps and check specifically around the documented
# shock at timestep 43.
sorted_by_f1 = sorted(per_timestep, key=lambda r: r["f1"])
worst_5 = sorted_by_f1[:5]
around_43 = [r for r in per_timestep if 40 <= r["timestep"] <= 46]
mean_f1_before = sum(r["f1"] for r in per_timestep if r["timestep"] < 43) / max(1, len([r for r in per_timestep if r["timestep"] < 43]))
mean_f1_after = sum(r["f1"] for r in per_timestep if r["timestep"] >= 43) / max(1, len([r for r in per_timestep if r["timestep"] >= 43]))

print(f"\nMean F1 before timestep 43: {mean_f1_before:.4f}")
print(f"Mean F1 from timestep 43 onward: {mean_f1_after:.4f}")
print(f"Worst 5 timesteps by F1: {[(r['timestep'], r['f1']) for r in worst_5]}")

set_drift_analysis({
    "description": "Per-timestep precision/recall/F1 for the baseline model (threshold 0.5), computed from data/predictions.pkl grouped by timestep - not a retrain, just a breakdown of the existing predictions. Checks for the concept drift documented in the Elliptic literature around timestep 43 (attributed to a dark web marketplace shutdown).",
    "per_timestep": per_timestep,
    "mean_f1_before_43": round(mean_f1_before, 4),
    "mean_f1_from_43": round(mean_f1_after, 4),
    "worst_timesteps": worst_5,
})
print("\nSaved to data/results.json")
