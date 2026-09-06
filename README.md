# Transaction Graph Fraud Detection

## Problem
[1 paragraph: connect this to your SCSC investigation background - why you built this]

## Dataset
Elliptic Data Set - a labeled Bitcoin transaction graph (203,769 nodes, 234,355 edges,
used in published anti-money-laundering research).

## Approach
1. Built a directed transaction graph from the edge list
2. Detected short cycles (2-4 hops) per timestep, flagging potential layering patterns
3. Ran Louvain community detection to identify tightly-connected wallet clusters
4. Computed degree and betweenness centrality to flag "mixer"-like addresses
5. Trained an XGBoost classifier comparing baseline features vs. graph-augmented features

## Results
Time-based split: trained on timesteps 1-34, tested on timesteps 35-49 (16,670 test transactions, 1,083 illicit).

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| Baseline | 0.90 | 0.73 | 0.80 |
| Graph-augmented | 0.95 | 0.73 | 0.82 |

Graph features raised precision on the illicit class by 5 points at the same recall, i.e. fewer false
positives on flagged transactions, without catching more or fewer of the actual illicit ones.

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
A graph neural network (e.g., EvolveGCN, which Elliptic was originally designed to
benchmark) could likely improve on these results by learning graph structure
end-to-end rather than using hand-engineered graph features.

## Running it
```
venv\Scripts\activate
python src\load_data.py
python src\build_graph.py
python src\cycle_detection.py
python src\community_detection.py
python src\centrality.py
python src\train_model.py
uvicorn src.api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive API, or `GET /analyze/{tx_id}` for a JSON risk score.
Dataset CSVs go in `data/` (gitignored) - see Part 4 of the setup guide for where to get them.
