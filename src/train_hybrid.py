"""
Hybrid model: GCN-learned embeddings + hand-engineered graph features +
raw features, fed into XGBoost.

train_model.py compares raw features vs. raw+hand-engineered graph features.
train_gnn.py compares raw features vs. a GCN that learns its own graph
representation. Neither uses BOTH kinds of graph signal together. This
script does: it takes the hidden-layer embedding from the already-trained
GCN (data/gcn_model.pt) - a 100-dim learned summary of each node's
neighborhood - and adds it alongside the same 4 hand-engineered features
used in train_model.py, on top of the 165 raw features. The idea: hand-
engineered features (cycle flag, community ratio, centrality) encode
specific, interpretable structural priors a human chose in advance, while
the GCN embedding can encode structural patterns nobody thought to
hand-craft. XGBoost then decides how much to weight each source.

Requires train_gnn.py to have already been run (needs data/gcn_model.pt).
"""
import pandas as pd
import numpy as np
import pickle
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score

from results_io import upsert_model

torch.manual_seed(42)
np.random.seed(42)

# --- Load data (same setup as train_gnn.py, for consistent node ordering) ---
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

df = features.merge(classes, on='txId', how='left')
tx_ids = df['txId'].values
id_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
N = len(tx_ids)

feat_cols = [c for c in df.columns if c.startswith('feat_')]
X = torch.tensor(df[feat_cols].values, dtype=torch.float32)

src = np.array([id_to_idx[t] for t in edges['txId1']])
dst = np.array([id_to_idx[t] for t in edges['txId2']])
row = np.concatenate([src, dst, np.arange(N)])
col = np.concatenate([dst, src, np.arange(N)])
deg = np.zeros(N)
np.add.at(deg, row, 1.0)
deg_inv_sqrt = 1.0 / np.sqrt(deg)
weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
values = torch.tensor(weight, dtype=torch.float32)
adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()


class GCN(nn.Module):
    """Must match train_gnn.py's architecture exactly to load its weights."""
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.3):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def hidden(self, x, adj):
        h = torch.sparse.mm(adj, x)
        return F.relu(self.w1(h))

    def forward(self, x, adj):
        h = self.hidden(x, adj)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        return self.w2(h)


gcn = GCN(in_dim=X.shape[1], hidden_dim=100)
gcn.load_state_dict(torch.load('data/gcn_model.pt'))
gcn.eval()

print("Extracting GCN hidden-layer embeddings (100-dim per node)...")
with torch.no_grad():
    embeddings = gcn.hidden(X, adj).numpy()
emb_cols = [f'gcn_emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
emb_df['txId'] = tx_ids

# --- Hand-engineered graph features (same as train_model.py) ---
df = df.merge(emb_df, on='txId', how='left')
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})

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

train = df[df['timestep'] <= 34]
test = df[df['timestep'] > 34]

hand_cols = ['fan_ratio', 'community_illicit_ratio', 'degree_centrality', 'betweenness_centrality']
hybrid_cols = feat_cols + hand_cols + emb_cols

X_train = train[hybrid_cols]
X_test = test[hybrid_cols]
y_train = train['class']
y_test = test['class']

print(f"Hybrid feature count: {len(hybrid_cols)} "
      f"({len(feat_cols)} raw + {len(hand_cols)} hand-engineered + {len(emb_cols)} GCN embedding)")

print("\n=== Hybrid model (raw + hand-engineered graph features + GCN embeddings, XGBoost) ===")
model_hybrid = XGBClassifier(eval_metric='logloss')
model_hybrid.fit(X_train, y_train)
preds_hybrid = model_hybrid.predict(X_test)
proba_hybrid = model_hybrid.predict_proba(X_test)[:, 1]
auc_pr_hybrid = average_precision_score(y_test, proba_hybrid)
report_hybrid = classification_report(y_test, preds_hybrid, target_names=['licit', 'illicit'],
                                       output_dict=True)
print(classification_report(y_test, preds_hybrid, target_names=['licit', 'illicit']))
print(f"AUC-PR (illicit): {auc_pr_hybrid:.4f}")

upsert_model("hybrid", {
    "name": "Hybrid",
    "description": "Raw + hand-engineered features (fan-in/fan-out ratio, community illicit ratio, degree/betweenness centrality) + GCN hidden-layer embeddings, XGBoost",
    "precision": round(report_hybrid['illicit']['precision'], 4),
    "recall": round(report_hybrid['illicit']['recall'], 4),
    "f1": round(report_hybrid['illicit']['f1-score'], 4),
    "auc_pr": round(float(auc_pr_hybrid), 4),
})

joblib.dump(model_hybrid, 'data/hybrid_model.pkl')
print("Model saved to data/hybrid_model.pkl")
print("Results written to data/results.json")
