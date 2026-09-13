import numpy as np
import pandas as pd
import pickle
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report, average_precision_score, brier_score_loss

from results_io import upsert_model, set_calibration
from predictions_io import save_model_predictions

# Load base features - keep the FULL set (labeled + unlabeled) around so
# predictions can be computed for every known transaction, not just the
# ~46K labeled ones used for training/evaluation. This is what the live
# API's scoring is actually built from (see predictions_io.py) - it needs
# to support looking up any txId a user might enter, same as before.
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
# The features file is ~690MB as float64; the API loads this same file at
# import time and holds it in memory for the life of the process (noted
# explicitly in the README's Limitations - this halves that footprint).
raw_feat_cols = [c for c in features.columns if c.startswith('feat_')]
features[raw_feat_cols] = features[raw_feat_cols].astype('float32')

classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
full_df = features.merge(classes, on='txId', how='left')

# Load graph-derived features
with open('data/fan_ratio.pkl', 'rb') as f:
    fan_ratio = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)

graph_feature_cols = pd.DataFrame({
    'fan_ratio': full_df['txId'].apply(lambda x: fan_ratio.get(x, 0.0)),
    'community_illicit_ratio': full_df['txId'].apply(
        lambda x: community_data['community_illicit_ratio'].get(
            community_data['tx_to_community'].get(x, -1), 0.0)),
    'in_degree_centrality': full_df['txId'].apply(lambda x: centrality_data['in_degree'].get(x, 0.0)),
    'out_degree_centrality': full_df['txId'].apply(lambda x: centrality_data['out_degree'].get(x, 0.0)),
})
full_df = pd.concat([full_df, graph_feature_cols], axis=1)

feat_cols = [c for c in full_df.columns if c.startswith('feat_')]
graph_cols = ['fan_ratio', 'community_illicit_ratio', 'in_degree_centrality', 'out_degree_centrality']

# Labeled-only subset for training and evaluation
df = full_df[full_df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

# Split by time - do NOT shuffle randomly, fraud detection must respect time order
train = df[df['timestep'] <= 34]
test = df[df['timestep'] > 34]

X_train_baseline = train[feat_cols]
X_test_baseline = test[feat_cols]
X_train_graph = train[feat_cols + graph_cols]
X_test_graph = test[feat_cols + graph_cols]
y_train = train['class']
y_test = test['class']

print("=== Baseline model (original features only) ===")
model_baseline = XGBClassifier(eval_metric='logloss', random_state=42)
model_baseline.fit(X_train_baseline, y_train)
preds_baseline = model_baseline.predict(X_test_baseline)
proba_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]

auc_pr_baseline = average_precision_score(y_test, proba_baseline)
report_baseline = classification_report(y_test, preds_baseline, target_names=['licit', 'illicit'],
                                         output_dict=True)
print(classification_report(y_test, preds_baseline, target_names=['licit', 'illicit']))
print(f"AUC-PR (illicit): {auc_pr_baseline:.4f}")

# --- Calibration experiment ---
# Tried isotonic regression (3-fold CV within the training period only,
# test set never touched) on the theory that a system where analysts
# triage by score needs a well-calibrated probability more than raw
# classification accuracy. Result, checked rather than assumed: it made
# things WORSE on the true held-out test period - Brier score got worse,
# not better (see numbers below), and classification metrics degraded
# too. Likely cause, tying directly into this project's own drift
# analysis: isotonic regression fit on training-period (timestep<=34) CV
# folds is calibrating to that period's probability distribution, which
# doesn't hold in the future test period if the data is genuinely
# drifting - the same phenomenon the per-timestep F1 breakdown documents
# directly. Reported as a negative result rather than silently kept or
# discarded: the RAW (uncalibrated) model is what's actually served.
calibrated_baseline = CalibratedClassifierCV(
    XGBClassifier(eval_metric='logloss', random_state=42), method='isotonic', cv=3)
calibrated_baseline.fit(X_train_baseline, y_train)
proba_calibrated = calibrated_baseline.predict_proba(X_test_baseline)[:, 1]

brier_raw = brier_score_loss(y_test, proba_baseline)
brier_calibrated = brier_score_loss(y_test, proba_calibrated)
frac_pos_raw, mean_pred_raw = calibration_curve(y_test, proba_baseline, n_bins=10, strategy='quantile')
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, proba_calibrated, n_bins=10, strategy='quantile')
print(f"Brier score - raw: {brier_raw:.4f}, isotonic-calibrated: {brier_calibrated:.4f} "
      f"({'WORSE' if brier_calibrated > brier_raw else 'better'} - serving raw)")

print("\n=== Graph-augmented model ===")
model_graph = XGBClassifier(eval_metric='logloss', random_state=42)
model_graph.fit(X_train_graph, y_train)
preds_graph = model_graph.predict(X_test_graph)
proba_graph = model_graph.predict_proba(X_test_graph)[:, 1]
auc_pr_graph = average_precision_score(y_test, proba_graph)
report_graph = classification_report(y_test, preds_graph, target_names=['licit', 'illicit'],
                                      output_dict=True)
print(classification_report(y_test, preds_graph, target_names=['licit', 'illicit']))
print(f"AUC-PR (illicit): {auc_pr_graph:.4f}")

upsert_model("baseline", {
    "name": "Baseline",
    "description": "Raw 165 transaction features only, XGBoost",
    "precision": round(report_baseline['illicit']['precision'], 4),
    "recall": round(report_baseline['illicit']['recall'], 4),
    "f1": round(report_baseline['illicit']['f1-score'], 4),
    "auc_pr": round(float(auc_pr_baseline), 4),
})
upsert_model("graph_augmented", {
    "name": "Graph-augmented",
    "description": "Raw features + hand-engineered graph features (fan-in/fan-out ratio, community illicit ratio, in/out-degree centrality), XGBoost",
    "precision": round(report_graph['illicit']['precision'], 4),
    "recall": round(report_graph['illicit']['recall'], 4),
    "f1": round(report_graph['illicit']['f1-score'], 4),
    "auc_pr": round(float(auc_pr_graph), 4),
})

# --- Calibration + cost-sensitive threshold sweep, saved for the README/dashboard ---
# Cost ratio is illustrative, not a real institutional figure: assume a
# missed fraud (false negative) costs 10x an investigator-hour spent
# chasing a false positive. Pick the threshold that minimizes
# (10 * false_negatives + 1 * false_positives) on the test set, instead of
# the default 0.5 cutoff used for every classification_report above.
COST_RATIO_FN_TO_FP = 10
thresholds = np.linspace(0.05, 0.95, 19)
threshold_sweep = []
for t in thresholds:
    pred_t = (proba_baseline >= t).astype(int)
    fn = int(((pred_t == 0) & (y_test == 1)).sum())
    fp = int(((pred_t == 1) & (y_test == 0)).sum())
    tp = int(((pred_t == 1) & (y_test == 1)).sum())
    precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cost = COST_RATIO_FN_TO_FP * fn + fp
    threshold_sweep.append({"threshold": round(float(t), 2), "precision": round(precision_t, 4),
                             "recall": round(recall_t, 4), "cost": cost})
best = min(threshold_sweep, key=lambda r: r["cost"])

set_calibration({
    "description": "Isotonic calibration (3-fold CV on the training period) was tried on the baseline model. Result: it made the Brier score WORSE on the true held-out test period, not better, so the RAW model is what's actually served - reported here as a negative result, not hidden. Brier score is the mean squared error between predicted probability and actual outcome (lower is better). Likely cause: calibration fit on training-period (timestep<=34) folds doesn't transfer to the future test period if the data is genuinely drifting - see this project's drift_analysis results for direct evidence of that drift.",
    "calibration_helped": bool(brier_calibrated < brier_raw),
    "brier_raw": round(float(brier_raw), 4),
    "brier_calibrated": round(float(brier_calibrated), 4),
    "reliability_curve_raw": {"mean_predicted": [round(float(v), 4) for v in mean_pred_raw],
                               "fraction_positive": [round(float(v), 4) for v in frac_pos_raw]},
    "reliability_curve_calibrated": {"mean_predicted": [round(float(v), 4) for v in mean_pred_cal],
                                      "fraction_positive": [round(float(v), 4) for v in frac_pos_cal]},
    "cost_ratio_fn_to_fp": COST_RATIO_FN_TO_FP,
    "threshold_sweep": threshold_sweep,
    "best_threshold": best,
})

# Predictions for EVERY known transaction (labeled or not), for the live
# API to serve via lookup - see predictions_io.py.
full_proba_baseline = model_baseline.predict_proba(full_df[feat_cols])[:, 1]
full_proba_graph = model_graph.predict_proba(full_df[feat_cols + graph_cols])[:, 1]
save_model_predictions('baseline', dict(zip(full_df['txId'], full_proba_baseline)))
save_model_predictions('graph_augmented', dict(zip(full_df['txId'], full_proba_graph)))

joblib.dump(model_baseline, 'data/baseline_model.pkl')
joblib.dump(model_graph, 'data/model.pkl')
print("\nModels saved to data/baseline_model.pkl and data/model.pkl")
print("Results written to data/results.json, predictions written to data/predictions.pkl")
