import os
import json
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

with open('data/fan_ratio.pkl', 'rb') as f:
    fan_ratio_data = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)

# Metrics live in data/results.json, written by the training scripts
# themselves (train_model.py, train_gnn.py, train_hybrid.py,
# cross_validate.py) via results_io.py - not hand-copied here, since that
# manual-transcription step is exactly how this project once shipped
# numbers that didn't match what was actually trained.
with open('data/results.json') as f:
    RESULTS = json.load(f)


def score_tx(tx_id: int):
    row = features[features['txId'] == tx_id]
    if row.empty:
        return {"tx_id": tx_id, "error": "Transaction ID not found"}

    fan_ratio_val = fan_ratio_data.get(tx_id, 0.0)
    community_ratio = community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(tx_id, -1), 0.0)
    degree = centrality_data['degree'].get(tx_id, 0.0)
    betweenness = centrality_data['betweenness'].get(tx_id, 0.0)

    # Build a one-row DataFrame with the same column names/order train_model.py
    # used, rather than a raw list - the model was fit on named columns, and a
    # positional list is one accidental reorder away from silently wrong
    # predictions with no warning.
    row_input = row[feat_cols].copy()
    row_input['fan_ratio'] = fan_ratio_val
    row_input['community_illicit_ratio'] = community_ratio
    row_input['degree_centrality'] = degree
    row_input['betweenness_centrality'] = betweenness

    risk_score = model.predict_proba(row_input)[0][1]

    return {
        "tx_id": tx_id,
        "risk_score": float(risk_score),
        "flags": {
            "pass_through_like": bool(fan_ratio_val >= 0.9),
            "high_risk_community": bool(community_ratio > 0.3)
        }
    }


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.get("/api")
def root():
    return {"message": "Fraud Risk Scoring API. Try /analyze/{tx_id}"}


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
