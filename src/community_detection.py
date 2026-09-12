import pickle
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

# Must match the train/test cutoff used in train_model.py / train_hybrid.py /
# api.py. community_illicit_ratio is a feature used for BOTH train and test
# rows, so it must only ever be computed from labels a real deployment would
# actually have at prediction time - i.e. training-period labels only.
# Computing it from all labels (including test-period ones) leaks the future
# into a feature: a caught-in-review bug, see README's "Methodology fix" note.
TRAIN_CUTOFF = 34

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

undirected_G = G.to_undirected()

print("Running community detection (this may take a few minutes)...")
communities = nx_comm.louvain_communities(undirected_G, seed=42)
print(f"Found {len(communities)} communities")

# tx_to_community depends only on graph structure (Louvain never looks at
# labels), so it's safe regardless of any train/test split.
tx_to_community = {}
for i, community in enumerate(communities):
    for tx in community:
        tx_to_community[tx] = i

# Load labels + timesteps so the illicit ratio can be restricted to the
# training period only.
features = pd.read_csv('data/elliptic_txs_features.csv', header=None, usecols=[0, 1])
features.columns = ['txId', 'timestep']
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
classes['class'] = classes['class'].map({'1': 1, '2': 0, 'unknown': None})
classes = classes.merge(features, on='txId', how='left')

train_txids = set(features[features['timestep'] <= TRAIN_CUTOFF]['txId'])

community_illicit_ratio = {}
for i, community in enumerate(communities):
    labels_in_community = classes[
        classes['txId'].isin(community) & classes['txId'].isin(train_txids)
    ]['class'].dropna()
    if len(labels_in_community) > 0:
        community_illicit_ratio[i] = labels_in_community.mean()
    else:
        community_illicit_ratio[i] = 0.0

with open('data/community_data.pkl', 'wb') as f:
    pickle.dump({'tx_to_community': tx_to_community,
                 'community_illicit_ratio': community_illicit_ratio,
                 'train_cutoff': TRAIN_CUTOFF}, f)

print(f"Community data saved (illicit ratio computed from timestep<={TRAIN_CUTOFF} labels only).")
