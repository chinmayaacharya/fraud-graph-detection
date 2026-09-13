import os
import json
import random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Fraud Risk Scoring API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Every model's per-transaction probability is precomputed by the training
# scripts (train_model.py, train_gnn.py, train_hybrid.py) into
# data/predictions.pkl - the API does a dict lookup, not live model
# inference. This avoids needing to load 3 different model formats (two
# XGBoost variants, a GCN needing a live adjacency matrix + torch) into
# this process, and avoids the exact class of bug this project already hit
# once: building a live feature vector by hand and getting a column out of
# order relative to what the model was trained on.
with open('data/predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

# NOTE on memory: this loads the full ~690MB features file into memory for
# the life of the process, just to check "does this txId exist" - fine for
# a local/demo deployment, worth knowing before running this anywhere with
# tighter memory limits. float32 (vs pandas' float64 default) roughly
# halves it; only txId is actually needed here, so this could be trimmed
# further (usecols=[0]) if memory ever becomes a real constraint.
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
raw_feat_cols = [c for c in features.columns if c.startswith('feat_')]
features[raw_feat_cols] = features[raw_feat_cols].astype('float32')
valid_tx_ids = set(features['txId'])

classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})

with open('data/fan_ratio.pkl', 'rb') as f:
    fan_ratio_data = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)

# Metrics live in data/results.json, written by the training scripts
# themselves via results_io.py - not hand-copied here, since that
# manual-transcription step is exactly how this project once shipped
# numbers that didn't match what was actually trained.
with open('data/results.json') as f:
    RESULTS = json.load(f)


def score_tx(tx_id: int):
    if tx_id not in valid_tx_ids:
        return {"tx_id": tx_id, "error": "Transaction ID not found"}

    scores = predictions.get(tx_id, {})

    fan_ratio_val = fan_ratio_data.get(tx_id, 0.0)
    community_ratio = community_data['community_illicit_ratio'].get(
        community_data['tx_to_community'].get(tx_id, -1), 0.0)

    return {
        "tx_id": tx_id,
        # The project's own evaluation (see /results) found the baseline
        # model has the best F1 of the four - so it's the headline score,
        # not graph-augmented, even though graph-augmented uses more of
        # the work in this project. risk_scores below gives the full
        # breakdown so nothing is hidden.
        "risk_score": scores.get("baseline"),
        "risk_scores": scores,
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
