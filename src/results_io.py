"""
Shared helper so every training script writes its own metrics straight
into data/results.json, instead of numbers being hand-copied into a
hardcoded dict in api.py after every rerun - which is exactly how this
project ended up shipping stale numbers before (a manual-transcription
step with no check that it was ever actually done).

Each script that computes a model's metrics calls upsert_model() with its
own results; api.py just loads and serves the file as-is via GET /results.
"""
import json
import os

RESULTS_PATH = 'data/results.json'

DEFAULT = {
    "test_set": {"total": 16670, "illicit": 1083},
    "models": [],
    "cross_validation": None,
    "note": ""
}


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return dict(DEFAULT)


def save_results(data):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def upsert_model(model_id, entry):
    """Insert or replace one model's results, keyed by model_id."""
    data = load_results()
    entry = {"id": model_id, **entry}
    data["models"] = [m for m in data.get("models", []) if m.get("id") != model_id] + [entry]
    save_results(data)


def set_cross_validation(cv_entry):
    data = load_results()
    data["cross_validation"] = cv_entry
    save_results(data)


def set_note(note_text):
    data = load_results()
    data["note"] = note_text
    save_results(data)
