"""
Walk-forward cross-validation across multiple time-based splits.

train_model.py reports a single train/test split (timestep <=34 / >34).
A single split can't tell you whether that result was a lucky cut or a
robust pattern, which matters for any claim made from it. This script
re-runs the baseline and graph-augmented XGBoost models across 3
different time-based splits (walk-forward validation - the correct
analog of k-fold CV for temporally ordered data, where a random shuffle
would leak future information) and reports mean +/- std for each metric.

Scope note: this deliberately excludes the GCN and hybrid models. Both
depend on a GCN that was trained once, using labels from timestep <=34
only. Reusing that same frozen GCN/embeddings for a CV split whose test
portion overlaps timestep <=34 would leak labels the GCN was fit on into
what's being called a "test" set. Properly cross-validating the GCN would
require retraining it from scratch for every split, which is a real
computational cost (multiple minutes each on CPU) left as future work
rather than done silently or incorrectly here.
"""
import pandas as pd
import pickle
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
df = features.merge(classes, on='txId', how='left')
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

with open('data/cycle_flags.pkl', 'rb') as f:
    cycle_flags = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)

df['in_cycle'] = df['txId'].apply(lambda x: 1 if x in cycle_flags else 0)
df['community_illicit_ratio'] = df['txId'].apply(
    lambda x: community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(x, -1), 0.0))
df['degree_centrality'] = df['txId'].apply(lambda x: centrality_data['degree'].get(x, 0.0))
df['betweenness_centrality'] = df['txId'].apply(lambda x: centrality_data['betweenness'].get(x, 0.0))

feat_cols = [c for c in df.columns if c.startswith('feat_')]
graph_cols = ['in_cycle', 'community_illicit_ratio', 'degree_centrality', 'betweenness_centrality']

# Walk-forward splits: (train up to and including this timestep, then test on the rest)
SPLITS = [26, 30, 34, 38]

results = {"baseline": {"precision": [], "recall": [], "f1": []},
           "graph_augmented": {"precision": [], "recall": [], "f1": []}}

for cutoff in SPLITS:
    train = df[df['timestep'] <= cutoff]
    test = df[df['timestep'] > cutoff]
    y_train, y_test = train['class'], test['class']

    print(f"\n--- Split: train timestep<={cutoff} ({len(train)} rows), "
          f"test timestep>{cutoff} ({len(test)} rows, {(y_test==1).sum()} illicit) ---")

    for name, cols, key in [
        ("Baseline", feat_cols, "baseline"),
        ("Graph-augmented", feat_cols + graph_cols, "graph_augmented"),
    ]:
        model = XGBClassifier(eval_metric='logloss')
        model.fit(train[cols], y_train)
        preds = model.predict(test[cols])
        report = classification_report(y_test, preds, target_names=['licit', 'illicit'],
                                        output_dict=True, zero_division=0)
        p, r, f1 = report['illicit']['precision'], report['illicit']['recall'], report['illicit']['f1-score']
        results[key]["precision"].append(p)
        results[key]["recall"].append(r)
        results[key]["f1"].append(f1)
        print(f"  {name:18s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

print("\n=== Walk-forward CV summary (mean +/- std across {} splits) ===".format(len(SPLITS)))
summary = {}
for key, label in [("baseline", "Baseline"), ("graph_augmented", "Graph-augmented")]:
    m = {stat: (float(np.mean(results[key][stat])), float(np.std(results[key][stat])))
         for stat in ["precision", "recall", "f1"]}
    summary[key] = m
    print(f"{label:18s} " + " ".join(
        f"{stat}={m[stat][0]:.3f}+/-{m[stat][1]:.3f}" for stat in ["precision", "recall", "f1"]))

with open('data/cv_results.json', 'w') as f:
    json.dump({"splits": SPLITS, "results": results, "summary": summary}, f, indent=2)
print("\nSaved to data/cv_results.json")
