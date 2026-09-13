"""
Shared helper for accumulating per-transaction predictions from all four
models into one data/predictions.pkl, so the live API can show every
model's score for a transaction via a plain dict lookup - no model
inference (and for the GCN, no torch import) needed in the API process at
all, and no risk of the column-order/feature-alignment bugs that come with
building a live feature vector by hand.

Each training script computes predict_proba (or, for the GCN, softmax
probabilities) over ALL transactions it has features for - not just the
test split - so the API can score any known txId, matching what the
original hand-built score_tx() supported.
"""
import pickle
import os

PREDICTIONS_PATH = 'data/predictions.pkl'


def load_predictions():
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_model_predictions(model_id, tx_id_to_prob):
    """Merge one model's {txId: probability} into the shared file."""
    all_preds = load_predictions()
    for tx_id, prob in tx_id_to_prob.items():
        all_preds.setdefault(int(tx_id), {})[model_id] = float(prob)
    with open(PREDICTIONS_PATH, 'wb') as f:
        pickle.dump(all_preds, f)
