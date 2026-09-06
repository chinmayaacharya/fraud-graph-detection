from fastapi import FastAPI
import joblib
import pandas as pd
import pickle

app = FastAPI(title="Fraud Risk Scoring API")

model = joblib.load('data/model.pkl')

features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]

with open('data/cycle_flags.pkl', 'rb') as f:
    cycle_flags = pickle.load(f)
with open('data/community_data.pkl', 'rb') as f:
    community_data = pickle.load(f)
with open('data/centrality_data.pkl', 'rb') as f:
    centrality_data = pickle.load(f)


@app.get("/")
def root():
    return {"message": "Fraud Risk Scoring API. Try /analyze/{tx_id}"}


@app.get("/analyze/{tx_id}")
def analyze(tx_id: int):
    row = features[features['txId'] == tx_id]
    if row.empty:
        return {"error": "Transaction ID not found"}

    feat_cols = [c for c in features.columns if c.startswith('feat_')]
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
            "high_risk_community": community_ratio > 0.3
        }
    }
