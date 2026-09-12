"""
Fan-in/fan-out ratio - replaces in_cycle as the "layering" signal.

cycle_detection.py found zero cycles across all 49 timesteps: Bitcoin's
UTXO model makes this transaction graph a DAG by construction, so short
cycles are structurally impossible and in_cycle was a constant-0 feature
contributing nothing. That negative result stays documented (and
cycle_detection.py still runs, to keep the evidence behind it
reproducible) - but it means the model needs a DAG-appropriate stand-in
for "this looks like layering."

The actual structural signature of layering in a DAG isn't a loop, it's a
chain of pass-through wallets: money arrives from one or few sources and
is forwarded to one or few destinations in roughly balanced measure,
unlike a normal merchant (many small payments in, few payouts out) or
exchange wallet (many-to-many, generally imbalanced). This ratio captures
exactly that:

    fan_ratio = min(in_degree, out_degree) / max(in_degree, out_degree)

Bounded [0, 1]. Close to 1 = balanced in/out (pass-through-like). Close
to 0 = skewed (hub, pure sink, or pure source). Pure graph structure, no
label information involved - same category as centrality, not community
detection's label-derived ratio.
"""
import pickle

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

print("Computing fan-in/fan-out ratio per node...")
fan_ratio = {}
for node in G.nodes():
    fan_in = G.in_degree(node)
    fan_out = G.out_degree(node)
    larger = max(fan_in, fan_out)
    fan_ratio[node] = (min(fan_in, fan_out) / larger) if larger > 0 else 0.0

with open('data/fan_ratio.pkl', 'wb') as f:
    pickle.dump(fan_ratio, f)

mean_ratio = sum(fan_ratio.values()) / len(fan_ratio)
pass_through = sum(1 for v in fan_ratio.values() if v >= 0.9)
print(f"Fan ratio computed for {len(fan_ratio)} nodes.")
print(f"Mean fan ratio: {mean_ratio:.4f}")
print(f"Nodes with fan_ratio >= 0.9 (strongly pass-through-like): {pass_through}")
print("Saved to data/fan_ratio.pkl")
