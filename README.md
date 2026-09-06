# Transaction Graph Fraud Detection

## Problem
Cybercrime and financial-fraud investigations increasingly run into cryptocurrency transactions that
launder proceeds through long chains of wallets designed to obscure the money trail. Manually tracing
these chains transaction-by-transaction doesn't scale, and looking at any single transaction in
isolation misses the patterns that actually indicate laundering — a wallet's position in the broader
transaction graph (who it clusters with, how central it is, whether it sits in a loop of transfers)
often carries more signal than the transaction's own attributes. This project tests whether adding
that graph structure as features to a standard classifier meaningfully improves fraud detection over
using transaction-level features alone.

<!-- TODO: replace this with your own framing if you want it to reference specific investigation
     experience (e.g. SCSC casework) rather than the generic version above. -->

## Dataset
Elliptic Data Set - a labeled Bitcoin transaction graph (203,769 nodes, 234,355 edges,
used in published anti-money-laundering research).

## Approach
1. Built a directed transaction graph from the edge list
2. Detected short cycles (2-4 hops) per timestep, flagging potential layering patterns
3. Ran Louvain community detection to identify tightly-connected wallet clusters
4. Computed degree and betweenness centrality to flag "mixer"-like addresses
5. Trained an XGBoost classifier comparing baseline features vs. graph-augmented (hand-engineered) features
6. Trained a Graph Convolutional Network (GCN, `src/train_gnn.py`) that learns a representation of
   each transaction directly from graph structure via message passing, instead of using hand-picked
   graph summary statistics - a 2-layer network (100 hidden units) over the raw 165 features,
   propagated through a symmetric-normalized adjacency matrix (Kipf & Welling, 2017), trained
   transductively over the full graph with loss/evaluation restricted to labeled nodes

## Results
Time-based split: trained on timesteps 1-34, tested on timesteps 35-49 (16,670 test transactions, 1,083 illicit).

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| Baseline (raw features, XGBoost) | 0.90 | 0.73 | 0.80 |
| Graph-augmented (raw + hand features, XGBoost) | 0.95 | 0.73 | 0.82 |
| GCN (raw features + learned graph structure) | 0.62 | 0.62 | 0.62 |

Hand-engineered graph features raised precision on the illicit class by 5 points over the baseline at
the same recall, i.e. fewer false positives on flagged transactions, without catching more or fewer of
the actual illicit ones.

The GCN, despite having access to the full graph structure and learning its own representations rather
than relying on 4 hand-picked numbers, **underperforms both XGBoost models**. This matches the original
Elliptic benchmark paper (Weber et al., 2019, "Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics"), where a similarly simple GCN was likewise beaten
by a Random Forest using hand-engineered features on this same dataset. The likely reasons, consistent
with that paper's own discussion: (1) a plain 2-layer GCN with mean-field message passing dilutes a
node's own signal by averaging it with many neighbors, which hurts on a graph with highly variable node
degree; (2) it has no notion of time, while transaction semantics are inherently temporal (this is
exactly the gap EvolveGCN, a follow-up to that paper, was built to close); (3) tree ensembles like
XGBoost handle this dataset's class imbalance and feature scale variation more gracefully out of the box
than a shallow GCN trained with a single global class-weighted loss. This is a genuine negative result
for the "just add a GNN" hypothesis, not a bug - the take-away isn't "GNNs are bad," it's "a bare-bones
GNN doesn't automatically beat a well-featured tree model without more architectural work (attention,
temporal structure, tuning) than this project implements."

## Limitations
- Cycle detection found **zero cycles in every one of the 49 timesteps**. This isn't a bug: Bitcoin's
  UTXO model makes the transaction graph a directed acyclic graph by construction (an output can't be
  spent before the transaction that creates it exists), so short-cycle detection as a layering signal
  doesn't apply to this dataset's structure. The `in_cycle` feature is therefore constant (always 0) and
  contributes nothing to the graph-augmented model — the precision gain above comes from the community
  and centrality features instead.
- Louvain community detection was run on the graph as a whole (not per-timestep) and found 312 communities.
- Betweenness centrality sampled (k=500) rather than exact, for performance.

## Future work
A temporal graph neural network (e.g. EvolveGCN, which Elliptic was originally designed to benchmark,
or a GCN with attention such as GAT) could likely close the gap seen here and surpass the XGBoost
models, since the plain GCN in this project has no mechanism for modeling how the graph evolves across
timesteps. Other directions: per-timestep community detection (this project ran it on the whole graph
at once), exact rather than sampled betweenness centrality, and hyperparameter tuning for all three
models (none were tuned beyond library defaults, so this comparison reflects architecture choice, not
maximum achievable performance for any of them).

## Running it
```
venv\Scripts\activate
python src\load_data.py
python src\build_graph.py
python src\cycle_detection.py
python src\community_detection.py
python src\centrality.py
python src\train_model.py
python src\train_gnn.py
uvicorn src.api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive API, or `GET /analyze/{tx_id}` for a JSON risk score.
Dataset CSVs go in `data/` (gitignored) - see Part 4 of the setup guide for where to get them.
