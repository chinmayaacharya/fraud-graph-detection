import pickle
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

undirected_G = G.to_undirected()

print("Running community detection (this may take a few minutes)...")
communities = nx_comm.louvain_communities(undirected_G, seed=42)
print(f"Found {len(communities)} communities")

# Load labels to compute how "illicit" each community is
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})
classes['class'] = classes['class'].map({'1': 1, '2': 0, 'unknown': None})

tx_to_community = {}
for i, community in enumerate(communities):
    for tx in community:
        tx_to_community[tx] = i

community_illicit_ratio = {}
for i, community in enumerate(communities):
    labels_in_community = classes[classes['txId'].isin(community)]['class'].dropna()
    if len(labels_in_community) > 0:
        community_illicit_ratio[i] = labels_in_community.mean()
    else:
        community_illicit_ratio[i] = 0.0

with open('data/community_data.pkl', 'wb') as f:
    pickle.dump({'tx_to_community': tx_to_community,
                 'community_illicit_ratio': community_illicit_ratio}, f)

print("Community data saved.")
