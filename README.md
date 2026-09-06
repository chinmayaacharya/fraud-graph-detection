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
| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| Baseline | [fill in] | [fill in] | [fill in] |
| Graph-augmented | [fill in] | [fill in] | [fill in] |

## Limitations
- Community detection run per-timestep due to compute constraints
- Betweenness centrality sampled (k=500) rather than exact, for performance
- [any others you hit]

## Future work
A graph neural network (e.g., EvolveGCN, which Elliptic was originally designed to
benchmark) could likely improve on these results by learning graph structure
end-to-end rather than using hand-engineered graph features.

## Running it
See setup instructions in this repo. API docs available at `/docs` once running.
