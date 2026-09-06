import pandas as pd
import pickle
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load base features
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
df = features.merge(classes, on='txId', how='left')
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

# Load graph-derived features
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

# Split by time - do NOT shuffle randomly, fraud detection must respect time order
train = df[df['timestep'] <= 34]
test = df[df['timestep'] > 34]

feat_cols = [c for c in df.columns if c.startswith('feat_')]
graph_cols = ['in_cycle', 'community_illicit_ratio', 'degree_centrality', 'betweenness_centrality']

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
print(classification_report(y_test, preds_baseline, target_names=['licit', 'illicit']))

print("\n=== Graph-augmented model ===")
model_graph = XGBClassifier(eval_metric='logloss')
model_graph.fit(X_train_graph, y_train)
preds_graph = model_graph.predict(X_test_graph)
print(classification_report(y_test, preds_graph, target_names=['licit', 'illicit']))

# Save the better model for the API
joblib.dump(model_graph, 'data/model.pkl')
print("\nModel saved to data/model.pkl")
