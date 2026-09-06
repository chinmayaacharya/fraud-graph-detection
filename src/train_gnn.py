"""
Graph Convolutional Network baseline for the Elliptic dataset.

Unlike train_model.py (which hands XGBoost 4 hand-engineered graph summary
statistics - in_cycle, community ratio, degree, betweenness), this script
lets the model learn a representation of each transaction directly from
graph structure via message passing, using only the 165 raw features as
input. This is the same style of model (and roughly the same setup) used
in the original Elliptic benchmark paper (Weber et al. 2019) and its
follow-ups (e.g. EvolveGCN).

Design notes:
- Transductive setting: message passing runs over the FULL graph (all
  203,769 nodes, labeled and unlabeled), because a GCN needs its
  neighbors' features to compute a node's representation regardless of
  whether that neighbor has a label. Only the LOSS and the reported
  metrics are restricted to labeled nodes, split by timestep exactly like
  train_model.py (train: timestep <= 34, test: timestep > 34). This uses
  future nodes' features during propagation but never their labels during
  training, which matches how this benchmark is evaluated in the
  published literature on this dataset.
- No PyTorch Geometric dependency: message passing is done manually with
  a sparse symmetric-normalized adjacency matrix (standard GCN
  propagation rule from Kipf & Welling 2017), which avoids PyG's finicky
  platform-specific wheel installation.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

torch.manual_seed(42)
np.random.seed(42)

# --- Load data ---
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

df = features.merge(classes, on='txId', how='left')

# Node index = row position in the features file (0..N-1)
tx_ids = df['txId'].values
id_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
N = len(tx_ids)

feat_cols = [c for c in df.columns if c.startswith('feat_')]
X = torch.tensor(df[feat_cols].values, dtype=torch.float32)

label_map = {'1': 1, '2': 0, 'unknown': -1}
y = torch.tensor(df['class'].map(label_map).values, dtype=torch.long)
timestep = torch.tensor(df['timestep'].values, dtype=torch.long)

train_mask = (timestep <= 34) & (y >= 0)
test_mask = (timestep > 34) & (y >= 0)
print(f"Train nodes (labeled, timestep<=34): {train_mask.sum().item()}")
print(f"Test nodes (labeled, timestep>34): {test_mask.sum().item()}")

# --- Build sparse symmetric-normalized adjacency with self-loops ---
src = np.array([id_to_idx[t] for t in edges['txId1']])
dst = np.array([id_to_idx[t] for t in edges['txId2']])

# Make undirected (GCN propagation is typically symmetric) and add self-loops
row = np.concatenate([src, dst, np.arange(N)])
col = np.concatenate([dst, src, np.arange(N)])

deg = np.zeros(N)
np.add.at(deg, row, 1.0)
deg_inv_sqrt = 1.0 / np.sqrt(deg)
weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
values = torch.tensor(weight, dtype=torch.float32)
adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

print(f"Graph: {N} nodes, {len(src)} directed edges (undirected + self-loops for propagation)")


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.3):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        h = torch.sparse.mm(adj, x)
        h = F.relu(self.w1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        return self.w2(h)


model = GCN(in_dim=X.shape[1], hidden_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Illicit is ~10% of labeled data - weight the minority class so the model
# doesn't just learn to predict "licit" for everything.
n_illicit = (y[train_mask] == 1).sum().item()
n_licit = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_licit / n_illicit], dtype=torch.float32)
print(f"Class weights [licit, illicit]: {class_weights.tolist()}")

EPOCHS = 120
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(X, adj)
    loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            test_out = model(X, adj)
            test_preds = test_out[test_mask].argmax(dim=1)
            test_f1_illicit = classification_report(
                y[test_mask].numpy(), test_preds.numpy(),
                target_names=['licit', 'illicit'], output_dict=True, zero_division=0
            )['illicit']['f1-score']
        print(f"Epoch {epoch:3d} | train loss {loss.item():.4f} | test illicit F1 {test_f1_illicit:.4f}")

model.eval()
with torch.no_grad():
    final_out = model(X, adj)
    final_preds = final_out[test_mask].argmax(dim=1)

print("\n=== GCN (graph-structure model, 165 raw features + learned propagation) ===")
print(classification_report(y[test_mask].numpy(), final_preds.numpy(),
                             target_names=['licit', 'illicit'], zero_division=0))

torch.save(model.state_dict(), 'data/gcn_model.pt')
print("Model saved to data/gcn_model.pt")
