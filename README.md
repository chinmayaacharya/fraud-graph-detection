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

## Methodology fix: label leakage in `community_illicit_ratio` (caught in code review)

An earlier version of this project computed `community_illicit_ratio` (each wallet cluster's fraction
of *known* illicit members) from **all** labeled transactions regardless of timestep. Since that ratio
is used as a feature for both training and test rows, this meant a test-period transaction's own
feature value was partly derived from test-period labels — including, for some rows, its own label —
and every walk-forward CV fold reused the same globally-computed ratio, so the leakage was identical
and undetected at every fold. This is exactly the kind of bug that produces a stable-looking, confident,
*wrong* result: the original numbers below (0.90 → 0.95 precision, holding at every CV split) looked
like solid evidence, and were actually measuring a model that had partial access to the answer key.

**The fix**: `community_illicit_ratio` is now computed only from labels with `timestep <= 34` (the
training cutoff) in `community_detection.py`, and `cross_validate.py` recomputes it fresh, per fold,
from only that fold's training-period labels (community *membership* — which cluster a node is in — is
pure graph structure with no label involved, so that part was never affected and is reused as-is).
`in_cycle` and centrality were never affected either: `in_cycle` is a static structural check with no
labels involved, and both centrality measures (`centrality.py`) are computed from graph connectivity
alone, using no label information at all - full-graph centrality is a defensible design choice about
*structure*, not a leakage risk.

**What changed once this was fixed:**

| Model | Precision (before fix) | Precision (after fix) | F1 (before) | F1 (after) |
|---|---|---|---|---|
| Baseline | 0.90 | 0.90 (unchanged - never used this feature) | 0.80 | 0.80 |
| Graph-augmented | 0.95 | **0.99** | 0.82 | **0.77** |
| Hybrid | 0.97 | **0.99** | 0.81 | **0.76** |

The leak didn't just inflate a number - it flipped the conclusion. Precision went *up* after the fix
(the leaked feature had actually been acting as noise diluting an otherwise very strong precision
signal), but recall collapsed (0.73 → 0.63 for graph-augmented), and **F1 for both graph-feature models
is now worse than the baseline's**, not better. The honest finding is the opposite of what was
originally reported: on this dataset, with this feature set, adding hand-engineered graph features
trades a large amount of recall for a small amount of precision and nets a worse F1 than raw features
alone.

## Results (corrected)
Time-based split: trained on timesteps 1-34, tested on timesteps 35-49 (16,670 test transactions, 1,083 illicit).

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| **Baseline (raw features, XGBoost)** | 0.90 | 0.73 | **0.80** |
| Graph-augmented (raw + hand features, XGBoost) | 0.99 | 0.63 | 0.77 |
| GCN (raw features + learned graph structure) | 0.62 | 0.62 | 0.62 |
| Hybrid (raw + hand features + GCN embeddings, XGBoost) | 0.99 | 0.62 | 0.76 |

**The baseline now has the best F1 of the three XGBoost variants.** Both graph-feature models push
precision to near-perfect (0.99) - almost every transaction they flag as illicit really is - but at the
cost of missing over a third of actual illicit transactions that the baseline would have caught. Whether
that tradeoff is "better" depends entirely on the deployment context (a triage queue with limited
investigator time might genuinely prefer very-high-precision-lower-recall; a system trying to catch as
much laundering as possible would not), but it is **not** the free, no-tradeoff improvement the
pre-fix numbers suggested, and reporting it as one would be wrong.

The GCN remains the weakest standalone model (0.62/0.62/0.62), unaffected by this bug since it never
used `community_illicit_ratio` or any hand-engineered feature - it only ever saw raw features and graph
structure. This matches the original Elliptic benchmark paper (Weber et al., 2019, "Anti-Money
Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"), where
a similarly simple GCN was likewise beaten by a Random Forest using hand-engineered features on this
same dataset. Likely reasons, consistent with that paper's discussion: (1) a plain 2-layer GCN with
mean-field message passing dilutes a node's own signal by averaging it with many neighbors; (2) it has
no notion of time, while transaction semantics are inherently temporal (the gap EvolveGCN was built to
close); (3) tree ensembles handle this dataset's class imbalance more gracefully out of the box than a
shallow GCN with a single global class-weighted loss.

The hybrid model (GCN embeddings + hand features + raw features) shows the same precision/recall
tradeoff as graph-augmented, slightly worse on both - concatenating the GCN's embeddings on top of an
already-leaky feature set didn't fix or meaningfully change the underlying problem.

### Robustness: walk-forward cross-validation (corrected)
`src/cross_validate.py` re-runs the baseline and graph-augmented models across 4 time-based splits
(train up to timestep 26, 30, 34, or 38; test on everything after), recomputing
`community_illicit_ratio` fresh per fold from that fold's training labels only:

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) |
|---|---|---|---|
| **Baseline** | 0.914 ± 0.025 | 0.726 ± 0.060 | **0.807 ± 0.033** |
| Graph-augmented | 0.988 ± 0.004 | 0.597 ± 0.038 | 0.743 ± 0.030 |

This is the same story, now confirmed as *consistent*, not a one-split artifact: graph-augmented loses
to baseline on F1 at **every one of the 4 splits** (0.759 vs 0.849, 0.751 vs 0.819, 0.771 vs 0.804, 0.692
vs 0.757), while its precision is both higher and far more stable (±0.004 vs baseline's ±0.025). The
walk-forward CV doesn't rescue the graph-augmented model's F1 - it confirms the tradeoff is real and
reproducible, which is exactly what cross-validation is for: distinguishing a genuine, repeatable effect
(here, the precision/recall tradeoff) from a lucky single split.

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
  contributes nothing to any model.
- Louvain community detection was run on the graph as a whole (not per-timestep) and found 312 communities.
  Community *membership* uses no label information (Louvain only looks at graph connectivity) - only the
  per-community illicit *ratio* uses labels, and that computation is now restricted to training-period
  labels only, per the Methodology fix above.
- **Centrality (degree and betweenness) is computed from the full graph's structure, across all
  timesteps, using no label information whatsoever** - stated explicitly here because it's the one
  hand-engineered feature that touches "future" graph structure (edges from timesteps after the training
  cutoff), unlike the label-based community ratio. This is a defensible design choice, not a leakage bug:
  in a real deployment, the transaction graph's topology (who has transacted with whom) is observable
  as it forms, and using the fuller graph for structural measures like centrality is standard practice
  in this literature - but it is a real assumption worth being explicit about, since it means centrality
  values for training-period nodes are informed by connections that hadn't happened yet at that time.
- Betweenness centrality sampled (k=500) rather than exact, for performance.

## Future work
- **Why does the community feature trade recall for precision so sharply?** Now that the leak is fixed,
  this is the actual open question this project raises: the corrected `community_illicit_ratio` is a
  much sparser, more conservative signal (only training-period labels contribute), and XGBoost appears
  to use it as a strong "veto" - understandable given the earlier version's high scores were partly an
  artifact, but worth investigating directly (e.g. feature-importance/SHAP analysis on the corrected
  model) rather than left as a hypothesis.
- **Class-weighting consistency**: the GCN training explicitly upweights the illicit class
  (`class_weights` in `train_gnn.py`) to counter the ~10:1 imbalance, but the XGBoost models here rely
  on defaults with no equivalent (e.g. `scale_pos_weight`). Adding it would make the three approaches
  more methodologically comparable, and might independently affect the precision/recall tradeoff
  reported above - deliberately left undone in this fix so the leak's effect could be isolated cleanly
  from any other change, but a natural next experiment.
- A temporal graph neural network (e.g. EvolveGCN, which Elliptic was originally designed to benchmark,
  or a GCN with attention such as GAT) could still improve on the GCN's standalone results, since it has
  no mechanism for modeling how the graph evolves across timesteps.
- Other directions: per-timestep community detection (this project ran it on the whole graph at once),
  exact rather than sampled betweenness centrality, hyperparameter tuning for all models (none were
  tuned beyond library defaults), and extending walk-forward cross-validation to the GCN and hybrid
  models by retraining the GCN per split instead of reusing one frozen copy.

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
