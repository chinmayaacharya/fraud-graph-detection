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
7. Built a hybrid model (`src/train_hybrid.py`) that extracts the GCN's hidden-layer embeddings and
   feeds them into XGBoost alongside the raw and hand-engineered features, testing whether a learned
   representation and hand-crafted statistics are complementary rather than redundant
8. Ran walk-forward cross-validation (`src/cross_validate.py`) across 4 time-based splits for the
   baseline and graph-augmented models, to check whether the single-split result generalizes

## Results
Time-based split: trained on timesteps 1-34, tested on timesteps 35-49 (16,670 test transactions, 1,083 illicit).

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| Baseline (raw features, XGBoost) | 0.90 | 0.73 | 0.80 |
| Graph-augmented (raw + hand features, XGBoost) | 0.95 | 0.73 | 0.82 |
| GCN (raw features + learned graph structure) | 0.62 | 0.62 | 0.62 |
| Hybrid (raw + hand features + GCN embeddings, XGBoost) | 0.97 | 0.70 | 0.81 |

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

**The hybrid model is not a clean win, and reporting it as one would be dishonest.** Adding the GCN's
learned embeddings on top of the hand-engineered features pushed precision to the highest of any model
(0.97) but recall dropped to 0.70 (from 0.73), landing F1 at 0.81 - essentially tied with, not better
than, the graph-augmented model's 0.82. The correct reading is that the GCN's embeddings shifted the
precision/recall tradeoff rather than adding clean predictive signal on top of what the hand-engineered
features already captured: with 269 features and only ~30K training rows, XGBoost likely has less to
gain from 100 more (correlated, since they come from the same underlying graph) dimensions than from a
qualitatively different signal. This is itself a useful finding - concatenating a representation from a
weaker model onto a stronger model's inputs is not guaranteed to help, and didn't here.

### Robustness: walk-forward cross-validation
A single train/test split can't distinguish a real effect from a lucky cut. `src/cross_validate.py`
re-runs the baseline and graph-augmented models across 4 time-based splits (train up to timestep 26,
30, 34, or 38; test on everything after) and reports mean ± std:

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| Baseline | 0.914 ± 0.025 | 0.726 ± 0.060 | 0.807 ± 0.033 |
| Graph-augmented | 0.945 ± 0.012 | 0.725 ± 0.052 | 0.819 ± 0.030 |

The graph-augmented model's precision advantage holds at **every one of the 4 splits**, not just the
one originally reported, and its precision variance (±0.012) is roughly half the baseline's (±0.025) -
the hand-engineered graph features make the model's precision both better and more stable across time,
which is a stronger and more defensible claim than a single-split result alone would support.

**Scope note**: this cross-validation deliberately excludes the GCN and hybrid models. Both depend on a
GCN trained once, using labels only from timestep ≤34. Reusing that same frozen GCN/embeddings for a CV
split whose test portion overlaps timestep ≤34 (e.g. the ≤26 or ≤30 splits) would leak labels the GCN
was fit on into what's being called a "test" set. Properly cross-validating the GCN would mean
retraining it from scratch per split (each run takes several minutes on CPU) - a real cost, left as
future work rather than done incorrectly here.

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
at once), exact rather than sampled betweenness centrality, hyperparameter tuning for all models (none
were tuned beyond library defaults, so this comparison reflects architecture choice, not maximum
achievable performance for any of them), and extending walk-forward cross-validation to the GCN and
hybrid models by retraining the GCN per split instead of reusing one frozen copy.

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
python src\train_hybrid.py
python src\cross_validate.py
uvicorn src.api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive API, or `GET /analyze/{tx_id}` for a JSON risk score.
Dataset CSVs go in `data/` (gitignored) - see Part 4 of the setup guide for where to get them.
