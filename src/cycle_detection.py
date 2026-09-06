import pickle
import pandas as pd
import networkx as nx
import itertools

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

# Elliptic has 49 timesteps - working on the full graph at once is too slow,
# so we detect cycles within each timestep's subgraph
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]

# Safety cap: a pathologically dense timestep could have combinatorially many
# short cycles. Stop counting past this many so one bad timestep can't hang
# the whole run - the exact count doesn't matter, only "in a cycle or not".
MAX_CYCLES_PER_TIMESTEP = 20000

flagged_tx = set()

for t in features['timestep'].unique():
    tx_ids_this_step = set(features[features['timestep'] == t]['txId'])
    subgraph = G.subgraph(tx_ids_this_step)

    # length_bound=4 means we look for short loops (2-4 hops) - typical of layering
    try:
        cycle_iter = nx.simple_cycles(subgraph, length_bound=4)
        cycles = list(itertools.islice(cycle_iter, MAX_CYCLES_PER_TIMESTEP))
    except Exception as e:
        print(f"Timestep {t} skipped: {e}")
        continue

    for cycle in cycles:
        flagged_tx.update(cycle)

    hit_cap = " (hit cap, more may exist)" if len(cycles) == MAX_CYCLES_PER_TIMESTEP else ""
    print(f"Timestep {t}: found {len(cycles)} cycles{hit_cap}")

print(f"\nTotal unique transactions flagged as being in a cycle: {len(flagged_tx)}")

# Save the flagged set for later use
with open('data/cycle_flags.pkl', 'wb') as f:
    pickle.dump(flagged_tx, f)
