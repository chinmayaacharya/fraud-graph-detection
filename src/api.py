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

    row_features = row[feat_cols].values[0].tolist()

    in_cycle = 1 if tx_id in cycle_flags else 0
    community_ratio = community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(tx_id, -1), 0.0)
    degree = centrality_data['degree'].get(tx_id, 0.0)
    betweenness = centrality_data['betweenness'].get(tx_id, 0.0)

    full_features = [row_features + [in_cycle, community_ratio, degree, betweenness]]
    risk_score = model.predict_proba(full_features)[0][1]

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


# Results from the offline train/test runs (train_model.py, train_gnn.py) -
# time-based split, timesteps 1-34 train / 35-49 test. Not recomputed at
# request time; update these if the models are retrained with changes.
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
            "precision": 0.95, "recall": 0.73, "f1": 0.82
        },
        {
            "name": "GCN",
            "description": "Raw features + learned graph structure (2-layer Graph Convolutional Network, no hand-engineered features)",
            "precision": 0.62, "recall": 0.62, "f1": 0.62
        },
        {
            "name": "Hybrid",
            "description": "Raw + hand-engineered features + GCN hidden-layer embeddings, XGBoost",
            "precision": 0.97, "recall": 0.70, "f1": 0.81
        }
    ],
    "note": (
        "The GCN underperforms both XGBoost models, consistent with the original Elliptic "
        "benchmark paper (Weber et al., 2019), where a similarly simple GCN also lost to a "
        "Random Forest with hand-engineered features. Likely causes: no temporal modeling "
        "(the graph evolves over 49 timesteps and this GCN has no notion of time), a plain "
        "2-layer architecture with no attention, and no hyperparameter tuning on any of the "
        "models. The Hybrid model (adding the GCN's embeddings on top of the hand-engineered "
        "features) is not a clean win either: it reaches the highest precision of any model "
        "(0.97) but recall drops to 0.70, landing F1 at 0.81 - essentially tied with, not "
        "better than, the graph-augmented model. Concatenating a weaker model's learned "
        "representation onto a stronger model's inputs shifted the precision/recall tradeoff "
        "rather than adding clean predictive signal."
    ),
    "cross_validation": {
        "description": "Walk-forward CV across 4 time-based splits (train <= timestep 26/30/34/38, test on the rest), for the two models that don't depend on a GCN pretrained on a fixed label split.",
        "models": [
            {"name": "Baseline", "precision_mean": 0.914, "precision_std": 0.025,
             "recall_mean": 0.726, "recall_std": 0.060, "f1_mean": 0.807, "f1_std": 0.033},
            {"name": "Graph-augmented", "precision_mean": 0.945, "precision_std": 0.012,
             "recall_mean": 0.725, "recall_std": 0.052, "f1_mean": 0.819, "f1_std": 0.030}
        ],
        "note": (
            "Graph-augmented's precision advantage over baseline held at every one of the 4 "
            "splits, and its precision variance (+/-0.012) is roughly half the baseline's "
            "(+/-0.025) - the hand-engineered graph features make precision both better and "
            "more stable across time, not just better on one lucky split. GCN and Hybrid are "
            "excluded here because both depend on a GCN trained once using labels from "
            "timestep <=34 only; reusing it for a CV split whose test set overlaps that range "
            "would leak labels into the test set."
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
