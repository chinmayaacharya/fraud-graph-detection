import os
import random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import pickle

app = FastAPI(title="Fraud Risk Scoring API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

model = joblib.load('data/model.pkl')

features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
feat_cols = [c for c in features.columns if c.startswith('feat_')]

classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})

with open('data/cycle_flags.pkl', 'rb') as f:
    cycle_flags = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)


def score_tx(tx_id: int):
    row = features[features['txId'] == tx_id]
    if row.empty:
        return {"tx_id": tx_id, "error": "Transaction ID not found"}

    in_cycle = 1 if tx_id in cycle_flags else 0
    community_ratio = community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(tx_id, -1), 0.0)
    degree = centrality_data['degree'].get(tx_id, 0.0)
    betweenness = centrality_data['betweenness'].get(tx_id, 0.0)

    # Build a one-row DataFrame with the same column names/order train_model.py
    # used, rather than a raw list - the model was fit on named columns, and a
    # positional list is one accidental reorder away from silently wrong
    # predictions with no warning.
    row_input = row[feat_cols].copy()
    row_input['in_cycle'] = in_cycle
    row_input['community_illicit_ratio'] = community_ratio
    row_input['degree_centrality'] = degree
    row_input['betweenness_centrality'] = betweenness

    risk_score = model.predict_proba(row_input)[0][1]

    return {
        "tx_id": tx_id,
        "risk_score": float(risk_score),
        "flags": {
            "in_cycle": bool(in_cycle),
            "high_risk_community": bool(community_ratio > 0.3)
        }
    }


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.get("/api")
def root():
    return {"message": "Fraud Risk Scoring API. Try /analyze/{tx_id}"}


# Results from the offline train/test runs (train_model.py, train_gnn.py,
# train_hybrid.py, cross_validate.py) - time-based split, timesteps 1-34
# train / 35-49 test. Not recomputed at request time; update these if the
# models are retrained with changes.
#
# CORRECTED numbers as of the community_illicit_ratio leakage fix (see
# README's "Methodology fix" section): the original version of this feature
# was computed from ALL labels regardless of timestep, leaking test-period
# labels into a feature used for both train and test rows. Fixed in
# community_detection.py / cross_validate.py to use training-period labels
# only. The corrected numbers REVERSE the original conclusion: baseline now
# has the best F1, not the graph-augmented model.
RESULTS = {
    "test_set": {"total": 16670, "illicit": 1083},
    "models": [
        {
            "name": "Baseline",
            "description": "Raw 165 transaction features only, XGBoost",
            "precision": 0.90, "recall": 0.73, "f1": 0.80
        },
        {
            "name": "Graph-augmented",
            "description": "Raw features + hand-engineered graph features (cycle flag, community illicit ratio, degree/betweenness centrality), XGBoost",
            "precision": 0.99, "recall": 0.63, "f1": 0.77
        },
        {
            "name": "GCN",
            "description": "Raw features + learned graph structure (2-layer Graph Convolutional Network, no hand-engineered features)",
            "precision": 0.62, "recall": 0.62, "f1": 0.62
        },
        {
            "name": "Hybrid",
            "description": "Raw + hand-engineered features + GCN hidden-layer embeddings, XGBoost",
            "precision": 0.99, "recall": 0.62, "f1": 0.76
        }
    ],
    "note": (
        "CORRECTED after fixing a label-leakage bug in community_illicit_ratio (see README). "
        "The baseline now has the BEST F1 of the three XGBoost variants - the graph-augmented "
        "and hybrid models push precision to near-perfect (0.99) but recall collapses to "
        "~0.62-0.63, netting a worse F1 than raw features alone. This reverses the original "
        "(leaky) finding that graph features were a clean improvement. The GCN was never "
        "affected by this bug (it uses no hand-engineered features) and remains the weakest "
        "standalone model, consistent with the original Elliptic benchmark paper (Weber et "
        "al., 2019), where a similarly simple GCN also lost to a Random Forest with "
        "hand-engineered features. Likely GCN causes: no temporal modeling, a plain 2-layer "
        "architecture with no attention, and no hyperparameter tuning on any model here."
    ),
    "cross_validation": {
        "description": "Walk-forward CV across 4 time-based splits (train <= timestep 26/30/34/38, test on the rest), for the two models that don't depend on a GCN pretrained on a fixed label split. community_illicit_ratio is recomputed fresh per fold from that fold's training labels only.",
        "models": [
            {"name": "Baseline", "precision_mean": 0.914, "precision_std": 0.025,
             "recall_mean": 0.726, "recall_std": 0.060, "f1_mean": 0.807, "f1_std": 0.033},
            {"name": "Graph-augmented", "precision_mean": 0.988, "precision_std": 0.004,
             "recall_mean": 0.597, "recall_std": 0.038, "f1_mean": 0.743, "f1_std": 0.030}
        ],
        "note": (
            "CORRECTED: graph-augmented now loses to baseline on F1 at every one of the 4 "
            "splits (not just on average) - this isn't a lucky single split, it's a consistent "
            "precision/recall tradeoff. Graph-augmented's precision is both higher and far more "
            "stable (+/-0.004 vs baseline's +/-0.025), but that stability doesn't translate into "
            "a better F1. GCN and Hybrid are excluded here because both depend on a GCN trained "
            "once using labels from timestep <=34 only; reusing it for a CV split whose test set "
            "overlaps that range would leak labels into the test set."
        )
    }
}


@app.get("/results")
def results():
    return RESULTS


@app.get("/analyze/{tx_id}")
def analyze(tx_id: int):
    return score_tx(tx_id)


class BatchRequest(BaseModel):
    tx_ids: list[int]


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    return [score_tx(tx_id) for tx_id in req.tx_ids]


@app.get("/sample")
def sample(n: int = 20):
    # Mix known illicit and licit transactions so a batch run actually shows
    # contrast in risk scores, instead of n random (mostly licit) txns.
    labeled = classes[classes['class'] != 'unknown']
    illicit_ids = labeled[labeled['class'] == '1']['txId'].tolist()
    licit_ids = labeled[labeled['class'] == '2']['txId'].tolist()

    n_illicit = min(n // 2, len(illicit_ids))
    n_licit = min(n - n_illicit, len(licit_ids))

    sampled = random.sample(illicit_ids, n_illicit) + random.sample(licit_ids, n_licit)
    random.shuffle(sampled)
    return {"tx_ids": [int(x) for x in sampled]}
