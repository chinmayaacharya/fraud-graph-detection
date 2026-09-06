import pandas as pd
import networkx as nx
import pickle

edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

# Build a directed graph: an edge means "txId1 sent to txId2"
G = nx.DiGraph()
G.add_edges_from(edges.values)

print("Number of nodes (transactions):", G.number_of_nodes())
print("Number of edges (connections):", G.number_of_edges())

# Save the graph so other scripts can reuse it without rebuilding
with open('data/graph.pkl', 'wb') as f:
    pickle.dump(G, f)

print("Graph saved to data/graph.pkl")
