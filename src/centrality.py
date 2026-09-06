import pickle
import networkx as nx

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

print("Computing degree centrality...")
degree_cent = nx.degree_centrality(G)

print("Computing betweenness centrality (sampled for speed)...")
# k=500 samples 500 nodes instead of all - full betweenness on 200k+ nodes is too slow
betweenness_cent = nx.betweenness_centrality(G, k=500, seed=42)

with open('data/centrality_data.pkl', 'wb') as f:
    pickle.dump({'degree': degree_cent, 'betweenness': betweenness_cent}, f)

print("Centrality data saved.")
