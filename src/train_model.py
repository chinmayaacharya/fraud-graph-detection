import pandas as pd
import pickle
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score

from results_io import upsert_model

# Load base features
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
df = features.merge(classes, on='txId', how='left')
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

# Load graph-derived features
with open('data/fan_ratio.pkl', 'rb') as f:
    fan_ratio = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)

df['fan_ratio'] = df['txId'].apply(lambda x: fan_ratio.get(x, 0.0))
df['community_illicit_ratio'] = df['txId'].apply(
    lambda x: community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(x, -1), 0.0))
df['degree_centrality'] = df['txId'].apply(lambda x: centrality_data['degree'].get(x, 0.0))
df['betweenness_centrality'] = df['txId'].apply(lambda x: centrality_data['betweenness'].get(x, 0.0))

# Split by time - do NOT shuffle randomly, fraud detection must respect time order
train = df[df['timestep'] <= 34]
test = df[df['timestep'] > 34]

feat_cols = [c for c in df.columns if c.startswith('feat_')]
graph_cols = ['fan_ratio', 'community_illicit_ratio', 'degree_centrality', 'betweenness_centrality']

X_train_baseline = train[feat_cols]
X_test_baseline = test[feat_cols]
X_train_graph = train[feat_cols + graph_cols]
X_test_graph = test[feat_cols + graph_cols]
y_train = train['class']
y_test = test['class']

print("=== Baseline model (original features only) ===")
model_baseline = XGBClassifier(eval_metric='logloss')
model_baseline.fit(X_train_baseline, y_train)
preds_baseline = model_baseline.predict(X_test_baseline)
proba_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]
auc_pr_baseline = average_precision_score(y_test, proba_baseline)
report_baseline = classification_report(y_test, preds_baseline, target_names=['licit', 'illicit'],
                                         output_dict=True)
print(classification_report(y_test, preds_baseline, target_names=['licit', 'illicit']))
print(f"AUC-PR (illicit): {auc_pr_baseline:.4f}")

print("\n=== Graph-augmented model ===")
model_graph = XGBClassifier(eval_metric='logloss')
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

# Save the graph-augmented model for the API
joblib.dump(model_graph, 'data/model.pkl')
print("\nModel saved to data/model.pkl")
print("Results written to data/results.json")
