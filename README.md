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
2. Detected short cycles (2-4 hops) per timestep, flagging potential layering patterns - found none
   (see Limitations); `src/fan_ratio.py` computes the DAG-appropriate replacement feature instead
3. Ran Louvain community detection to identify tightly-connected wallet clusters
4. Computed in-degree and out-degree centrality separately (consolidation vs. dispersal signatures)
   to flag "mixer"-like addresses; betweenness centrality was tried, tested for stability, and dropped
   (see Limitations)
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
9. Every training script writes its own metrics into `data/results.json` (via `src/results_io.py`)
   rather than numbers being hand-copied into the API - `api.py` just loads and serves that file
10. Isotonic-calibrated the baseline model's probabilities (3-fold CV within the training period only)
    and ran a cost-sensitive threshold sweep, since the API's `risk_score` is meant to be usable for
    real triage, not just a classification-accuracy number
11. Computed per-timestep F1 across all 49 timesteps (`src/drift_analysis.py`) to check for concept
    drift - Elliptic is known to have a real distribution shift around timestep 43 (a dark web market
    shutdown), which a single train/test split entirely hides

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

**Update**: `in_cycle` (visible in the table above) has since been replaced by `fan_ratio` - see
Limitations below for why, and the Results section for the current feature set. Replacing it left these
particular numbers essentially unchanged (to 2 decimal places), which is itself informative: `in_cycle`
was already contributing nothing (it was constant), so removing it and adding a real structural feature
in its place had to be evaluated on its own merits rather than assumed to help - and it did not move the
needle much, an honest result reported in full below rather than quietly omitted.

## Results (current)
Time-based split: trained on timesteps 1-34, tested on timesteps 35-49 (16,670 test transactions, 1,083 illicit).
Graph features here are `fan_ratio` (replacing `in_cycle` - see Limitations), `community_illicit_ratio`
(training-period labels only), and in/out-degree centrality (betweenness was dropped - see Limitations).
Alongside precision/recall/F1 (which
depend on the model's default 0.5 classification threshold), **AUC-PR** (area under the
precision-recall curve) is reported too, since it measures a model's overall ability to *rank*
transactions by risk independent of any threshold choice - important here, because the two metrics tell
different parts of the story, as the discussion below explains.

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) | AUC-PR (illicit) |
|---|---|---|---|---|
| **Baseline (raw features, XGBoost)** | 0.90 | 0.73 | **0.80** | **0.80** |
| Graph-augmented (raw + hand features, XGBoost) | 0.99 | 0.63 | 0.77 | 0.79 |
| GCN (raw features + learned graph structure) | 0.33 | 0.69 | 0.45 | 0.45 |
| Hybrid (raw + hand features + GCN embeddings, XGBoost) | 0.98 | 0.67 | 0.79 | 0.79 |

**The baseline has the best F1 of the four models, but AUC-PR tells a more nuanced story for the two
XGBoost variants specifically.** By F1, both graph-feature models lose to the baseline: they push
precision to near-perfect (0.98-0.99) - almost every transaction they flag as illicit really is - but
miss noticeably more actual illicit transactions than the baseline would have caught. By AUC-PR, though,
baseline (0.80) and graph-augmented (0.79) are **close**, meaning the graph-augmented model's underlying
ability to rank risky transactions above safe ones is nearly as good as the baseline's - its F1
disadvantage comes largely from its default 0.5 probability threshold landing at a high-precision/
low-recall operating point, not from the model being fundamentally worse at the task (see the Calibration
section below for a threshold that does better than 0.5 for either model). Whether the high-precision
operating point is "better" still depends on deployment context (a triage queue with limited investigator
time might prefer it; a system trying to catch as much laundering as possible would not) - but it is
**not** the free, no-tradeoff improvement the pre-leak-fix numbers suggested.

**The GCN's numbers dropped substantially from an earlier version of this project** (was 0.62 across the
board, now 0.33 precision / 0.69 recall / 0.45 F1) - not because of a modeling change, but because of a
methodology fix with a real cost. The GCN's training loop used to print test-set F1 every 20 epochs
purely to monitor progress; no epoch count or hyperparameter was ever actually chosen based on that
number, but it looked like the test set was being watched during training, which a reviewer reasonably
flagged. The fix: training now uses timestep ≤30 (not ≤34) with 31-34 carved out as a genuine validation
split for that monitoring, and the true test set (>34) is touched exactly once, after training
completes. This is more defensible - but it also means the GCN trained on ~3,000 fewer labeled examples,
and empirically that cost it real performance. This is reported as-is rather than reverted to get a
better-looking number: a data-hungry model losing a meaningful chunk of its already-small labeled
training set is itself an honest, informative result about how little labeled data this benchmark
actually provides.

The GCN remains the weakest standalone model, unaffected by the community-ratio leakage bug (it never
used `community_illicit_ratio` or any hand-engineered feature - only raw features and graph structure).
This direction - GCN underperforming feature-engineered tree models - matches the original Elliptic
benchmark paper (Weber et al., 2019, "Anti-Money Laundering in Bitcoin: Experimenting with Graph
Convolutional Networks for Financial Forensics"), where a similarly simple GCN was likewise beaten by a
Random Forest using hand-engineered features on this same dataset. Likely reasons, consistent with that
paper's discussion: (1) a plain 2-layer GCN with mean-field message passing dilutes a node's own signal
by averaging it with many neighbors; (2) it has no notion of time, while transaction semantics are
inherently temporal (the gap EvolveGCN was built to close - see Future work); (3) tree ensembles handle
this dataset's class imbalance more gracefully out of the box than a shallow GCN with a single global
class-weighted loss; (4) now compounded by less training data after the validation-split fix above.

The hybrid model (GCN embeddings + hand features + raw features) shows a similar precision/recall
tradeoff to graph-augmented, with comparable AUC-PR (0.79) - concatenating the (now weaker) GCN's
embeddings on top of the hand-engineered features still didn't add clean predictive signal beyond what
the hand-engineered features alone provide.

### Robustness: walk-forward cross-validation (current)
`src/cross_validate.py` re-runs the baseline and graph-augmented models across 4 time-based splits
(train up to timestep 26, 30, 34, or 38; test on everything after), recomputing
`community_illicit_ratio` fresh per fold from that fold's training labels only:

| Model | Precision (illicit) | Recall (illicit) | F1 (illicit) | AUC-PR (illicit) |
|---|---|---|---|---|
| **Baseline** | 0.914 ± 0.025 | 0.726 ± 0.060 | **0.807 ± 0.033** | 0.829 ± 0.043 |
| Graph-augmented | 0.987 ± 0.005 | 0.607 ± 0.042 | 0.751 ± 0.033 | 0.827 ± 0.031 |

This confirms the single-split story is *consistent*, not an artifact: graph-augmented loses to baseline
on F1 at **every one of the 4 splits**, while its precision is both higher and far more stable (±0.005 vs
baseline's ±0.025). AUC-PR is close between the two models across every fold too (mean 0.829 vs 0.827,
baseline ahead on most splits) - reinforcing that the F1 gap reflects a threshold-specific tradeoff
rather than a fundamental difference in what each model has learned. The walk-forward CV doesn't rescue
the graph-augmented model's F1, but it does confirm the AUC-PR near-tie is real and reproducible, not a
lucky single split.

## Concept drift: the model doesn't fail gracefully, it falls off a cliff at timestep 43

Every number above is a single aggregate over the whole test period (timesteps 35-49). That average
hides something dramatic. `src/drift_analysis.py` breaks the baseline model's performance down by
individual timestep instead of averaging over 15 of them at once - Elliptic is documented in the
literature as having a real distribution shift around timestep 43, widely attributed to a dark web
marketplace shutdown changing the mix of transaction patterns, and this project's own model reproduces
that shock directly, not just as a citation:

| Test timestep | Illicit count | Precision | Recall | F1 |
|---|---|---|---|---|
| 35 | 182 | 0.96 | 0.97 | 0.96 |
| 36 | 33 | 0.74 | 0.97 | 0.84 |
| 37 | 40 | 1.00 | 0.68 | 0.81 |
| 38 | 111 | 0.97 | 0.90 | 0.94 |
| 39 | 81 | 0.94 | 0.93 | 0.93 |
| 40 | 112 | 0.92 | 0.65 | 0.76 |
| 41 | 116 | 0.97 | 0.94 | 0.96 |
| 42 | 239 | 0.95 | 0.80 | 0.87 |
| **43** | 24 | **0.00** | **0.00** | **0.00** |
| 44 | 24 | 0.06 | 0.04 | 0.05 |
| 45 | 5 | 0.00 | 0.00 | 0.00 |
| 46 | 2 | 0.10 | 0.50 | 0.17 |
| 47 | 22 | 0.00 | 0.00 | 0.00 |
| 48 | 36 | 0.50 | 0.03 | 0.05 |
| 49 | 56 | 0.14 | 0.02 | 0.03 |

**Mean F1 for timesteps 35-42: 0.90. Mean F1 for timesteps 43-49: 0.04.** This isn't a gradual decline -
it's a cliff, and it happens at exactly one timestep. Checked directly, not just inferred from the
aggregate: illicit transactions at timestep 42 get a mean predicted probability of 0.80 (correctly
high); at timestep 43, illicit transactions get a mean predicted probability of 0.008 - statistically
indistinguishable from licit transactions' 0.007. The model doesn't get *worse* at recognizing illicit
transactions after the shock, it loses essentially all discriminative signal for them, instantly.

The single test-period F1 of 0.80 reported everywhere else in this README is real, but it is an average
of "works very well" (0.90) and "doesn't work at all" (0.04), and that average is a genuinely misleading
number for anyone deciding whether to trust this model going forward in time. A system built on this
model and deployed at timestep 34 would have looked great for two months and then silently stopped
catching fraud entirely - while a naive monitoring setup watching only a rolling aggregate F1 might not
notice for a while, since the pre-shock timesteps are still in the average. This is arguably the single
most important finding in this project: not which feature set wins, but that *any* of these models can
fail this completely, this suddenly, on this exact dataset - a materially different, more useful
finding for a real deployment than the model comparison table above.

## Calibration and cost-sensitive thresholding

Precision/recall/F1 all evaluate the model's raw yes/no classification at the default 0.5 probability
threshold. Neither actually asks whether the *probability itself* means anything, or whether 0.5 is
the right cutoff for how this would really be used - two questions that matter more than raw accuracy
for a system where an analyst triages by score.

**Calibration - tried, and it made things worse.** Isotonic regression (3-fold CV within the training
period, test set never touched) was fit on top of the baseline model to check whether its probabilities
are well-calibrated (does "70% risk" really mean ~70% of such transactions are illicit?). Brier score
(mean squared error between predicted probability and outcome, lower is better) went from **0.0212
(raw) to 0.0284 (calibrated) - calibration made it worse**, not better, on the true held-out test
period. The likely cause connects directly to the drift finding above: isotonic regression fit on
training-period (≤34) folds is calibrating to that period's probability distribution, which does not
hold after the shock at timestep 43. The raw, uncalibrated model is what's actually served - this is
reported as a negative result rather than quietly dropped, because "we tried the standard fix and it
didn't work, here's why" is itself informative given the drift finding.

**Cost-sensitive thresholding - a small, real improvement.** The default 0.5 cutoff isn't chosen for
any principled reason. Assuming (illustratively, not from a real institutional cost model) that a missed
fraud costs 10x an investigator-hour spent on a false positive, sweeping thresholds from 0.05 to 0.95
against the baseline model's raw probabilities finds **0.4** minimizes total cost - not 0.5. At that
threshold: precision 0.87, recall 0.735 (vs. 0.90/0.73 at the default) - a modest, genuine gain in
recall at a small precision cost, which is the right direction to move if missing fraud is actually more
costly than chasing a false lead. Different cost assumptions would move this threshold; the sweep itself
(saved in `data/results.json`) is the reusable part, not this specific number.

**Scope note**: this cross-validation deliberately excludes the GCN and hybrid models. Both depend on a
GCN trained once, using labels only from timestep ≤34. Reusing that same frozen GCN/embeddings for a CV
split whose test portion overlaps timestep ≤34 (e.g. the ≤26 or ≤30 splits) would leak labels the GCN
was fit on into what's being called a "test" set. Properly cross-validating the GCN would mean
retraining it from scratch per split (each run takes several minutes on CPU) - a real cost, left as
future work rather than done incorrectly here.

## Serving predictions: which model does the live API actually use?

An earlier version of this project served predictions from the graph-augmented model - the same one
this README now shows loses to the baseline on F1. That's a real inconsistency (why deploy the model
your own evaluation says is worse?), caught in review and fixed: `train_model.py`, `train_gnn.py`, and
`train_hybrid.py` each precompute their model's probability for *every* known transaction (not just the
test split) and write it into `data/predictions.pkl` (`src/predictions_io.py`). The API
(`GET /analyze/{tx_id}`) does a plain dict lookup - no live model inference, no risk of rebuilding a
feature vector with columns in the wrong order (the exact bug class this project already hit once) -
and returns:
- `risk_score`: the **baseline** model's probability, since it has the best F1 of the four
- `risk_scores`: all four models' probabilities together, so the comparison this whole project is about
  is visible on every single request, not just in this README

This also means the API can score *any* known transaction, labeled or not (203,769 of them), not just
the ~46K used for training/evaluation - a strict improvement over the earlier version, which could only
build features for rows it could compute community/centrality lookups for at request time anyway.

## Limitations
- **The GCN trains on timestep ≤30 only (not ≤34)**, holding 31-34 out as a genuine validation split so
  training-time monitoring never touches the true test set (>34) - fixed after review flagged that the
  old setup printed test-set F1 during training, which wasn't actually used for any decision but looked
  like it could have been. This means the GCN (and the hybrid model, which uses its embeddings) has
  ~3,000 fewer labeled training examples than the two XGBoost models, which still train on the full
  ≤34 period - the four models in the results table are not working from identical amounts of data,
  and the GCN/hybrid numbers should be read with that in mind, not as a perfectly even comparison.
- Every result and finding in this README (including the concept-drift and calibration sections) is
  specific to this exact time-based split and this exact 49-timestep dataset. A different cutoff, a
  different fraud dataset, or a live deployment with continuously arriving data would need its own
  drift and calibration checks - none of this transfers by assumption.
- Cycle detection (`src/cycle_detection.py`) found **zero cycles in every one of the 49 timesteps**.
  This isn't a bug: Bitcoin's UTXO model makes the transaction graph a directed acyclic graph by
  construction (an output can't be spent before the transaction that creates it exists), so short-cycle
  detection as a layering signal doesn't apply to this dataset's structure. The script still runs and
  the negative result stays documented here for reproducibility, but its output (`in_cycle`, always 0)
  has been **replaced** in every model by `fan_ratio` (`src/fan_ratio.py`) - the min(in-degree,
  out-degree)/max(in-degree, out-degree) ratio for each wallet, which is the actual DAG-appropriate
  structural signature of layering: a chain of pass-through wallets receiving and forwarding money in
  roughly balanced amounts, as opposed to a normal merchant or exchange wallet's typically skewed
  in/out pattern. Worth being honest about its own limitation too: 76,046 of 203,769 nodes (37%) score
  ≥0.9 on this ratio, so "balanced in/out" is common in this graph, not a rare smoking gun - and,
  consistent with that, swapping it in for the useless `in_cycle` left the reported metrics essentially
  unchanged (see the Methodology fix section's "Update" note). It's a real, non-constant, defensible
  feature, but not shown here to be a strong one on its own.
- Louvain community detection was run on the graph as a whole (not per-timestep) and found 312 communities.
  Community *membership* uses no label information (Louvain only looks at graph connectivity) - only the
  per-community illicit *ratio* uses labels, and that computation is now restricted to training-period
  labels only, per the Methodology fix above. **Worth saying explicitly, since a reviewer will ask**:
  even with that fix, a test-period node in a community whose training-period members are mostly illicit
  still inherits a strong signal from those training labels - that's the intended mechanism (guilt by
  association), not a bug, and it's a standard *transductive* assumption (the whole graph's structure,
  including test-period edges, is known at feature-computation time; only test-period *labels* are
  withheld). It does mean this feature would behave differently in a true online/streaming deployment,
  where a brand-new community with no established members yet would have no signal to inherit.
- Betweenness centrality was computed at k=500 for earlier versions of this project and used as a
  feature; it was **dropped** after directly testing its stability (see `centrality.py`'s docstring) -
  two runs with different random seeds gave Spearman rank correlation of only 0.33 (Pearson 0.70, and
  96% of nodes scored exactly zero either way), meaning the feature was mostly reflecting which 500
  nodes happened to get sampled, not real structural signal. Raising `k` enough to fix this was tested
  and found impractical within reasonable runtime on a 200K+ node graph on a single machine - dropping
  it was the honest choice given that constraint. Degree centrality is now split into in-degree and
  out-degree separately (previously a single combined number) since they mean different things for
  fraud - many inputs suggests consolidation, many outputs suggests dispersal - and networkx's
  `degree_centrality()` on a directed graph silently combines both into one number if you're not
  careful to ask for them separately.
- **Centrality (in/out-degree) and `fan_ratio` are both computed from the full graph's
  structure, across all timesteps, using no label information whatsoever** - stated explicitly here
  because these are the hand-engineered features that touch "future" graph structure (edges from
  timesteps after the training cutoff), unlike the label-based community ratio, which is now
  restricted to training-period labels. This is a defensible design choice, not a leakage bug: in a
  real deployment, the transaction graph's topology (who has transacted with whom) is observable as it
  forms, and using the fuller graph for structural measures is standard practice in this literature
  (the GCN's message passing makes the same choice, for the same reason - see `train_gnn.py`'s design
  notes) - but it is a real assumption worth being explicit about, since it means these three features'
  values for training-period nodes are informed by connections that hadn't happened yet at that time.
- Betweenness centrality sampled (k=500) rather than exact, for performance.

## Future work
- **Find and use a better classification threshold for the graph-augmented model.** AUC-PR shows its
  ranking ability is on par with the baseline, so the F1 gap is a threshold artifact, not a fundamental
  weakness - the natural next step is a precision-recall curve analysis to pick a threshold that
  recovers more balanced recall, instead of relying on the default 0.5 cutoff used throughout this
  project for comparability.
- **Class-weighting consistency**: the GCN training explicitly upweights the illicit class
  (`class_weights` in `train_gnn.py`) to counter the ~10:1 imbalance, but the XGBoost models here rely
  on defaults with no equivalent (e.g. `scale_pos_weight`). Adding it would make the three approaches
  more methodologically comparable, and might independently affect the precision/recall tradeoff
  reported above - deliberately left undone here so each fix's effect could be isolated cleanly, but a
  natural next experiment.
- **`fan_ratio` is a weak feature on its own** (see Limitations) - worth trying combined with timestep-
  local statistics (e.g. a wallet's fan ratio relative to its timestep's distribution) rather than a
  single global value, or explicitly interacted with centrality in the model rather than left for
  XGBoost to discover.
- **A temporal graph neural network (EvolveGCN, or a GRU over per-timestep embeddings) is the
  highest-value addition this project doesn't yet have.** The plain GCN here has no notion of time; a
  temporal model is exactly what the literature (and this project's own drift analysis) points to as
  the natural next step, and could plausibly close the gap with the XGBoost models. Deliberately not
  attempted in this pass: it's a real multi-day implementation project on its own, not something to
  bolt on quickly alongside a bug-fix/audit pass without risking the same kind of rushed mistake this
  audit was fixing.
- **SHAP-based per-transaction explanations.** The dashboard shows a risk score but not *why* -
  investigators triaging real cases need to know which features drove a specific flag. Adding SHAP
  values for the served baseline model, surfaced per-transaction in the API/dashboard, would make it
  genuinely usable by an analyst rather than just a number.
- **Extending calibration and cost-sensitive thresholding to the other three models**, and revisiting
  the illustrative 10:1 false-negative-to-false-positive cost ratio used in the threshold sweep with a
  real institutional estimate if this were ever used for actual triage - the 10:1 figure here is a
  stand-in to demonstrate the method, not a researched number.
- **Checking whether the drift found around timestep 43 (see Results) affects graph-augmented, GCN, and
  hybrid the same way it affects the baseline** - this project's drift analysis only covers the served
  baseline model; it's plausible the graph features are more or less fragile around that shock than raw
  features alone.
- Other directions: per-timestep community detection (this project ran it on the whole graph at once),
  hyperparameter tuning for all models (none were tuned beyond library defaults), and extending
  walk-forward cross-validation to the GCN and hybrid models by retraining the GCN per split instead of
  reusing one frozen copy.

## Running it
```
venv\Scripts\activate
python src\load_data.py
python src\build_graph.py
python src\cycle_detection.py
python src\fan_ratio.py
python src\community_detection.py
python src\centrality.py
python src\train_model.py
python src\train_gnn.py
python src\train_hybrid.py
python src\cross_validate.py
python src\drift_analysis.py
uvicorn src.api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive API, or `GET /analyze/{tx_id}` for a JSON risk score.
Dataset CSVs go in `data/` (gitignored) - see Part 4 of the setup guide for where to get them.
