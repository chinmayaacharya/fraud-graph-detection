import pandas as pd
import pickle
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score

from results_io import upsert_model
from predictions_io import save_model_predictions

# Load base features - keep the FULL set (labeled + unlabeled) around so
# predictions can be computed for every known transaction, not just the
# ~46K labeled ones used for training/evaluation. This is what the live
# API's scoring is actually built from (see predictions_io.py) - it needs
# to support looking up any txId a user might enter, same as before.
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
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
    'degree_centrality': full_df['txId'].apply(lambda x: centrality_data['degree'].get(x, 0.0)),
    'betweenness_centrality': full_df['txId'].apply(lambda x: centrality_data['betweenness'].get(x, 0.0)),
})
full_df = pd.concat([full_df, graph_feature_cols], axis=1)

feat_cols = [c for c in full_df.columns if c.startswith('feat_')]
graph_cols = ['fan_ratio', 'community_illicit_ratio', 'degree_centrality', 'betweenness_centrality']

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
    "description": "Raw features + hand-engineered graph features (fan-in/fan-out ratio, community illicit ratio, degree/betweenness centrality), XGBoost",
    "precision": round(report_graph['illicit']['precision'], 4),
    "recall": round(report_graph['illicit']['recall'], 4),
    "f1": round(report_graph['illicit']['f1-score'], 4),
    "auc_pr": round(float(auc_pr_graph), 4),
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
