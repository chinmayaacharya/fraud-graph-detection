"""
Degree centrality (split by direction) and the betweenness decision.

Split in/out degree centrality: nx.degree_centrality() on a DiGraph
returns a single number per node based on total (in+out) degree, which
conflates two semantically different fraud signals - many inputs
(consolidation, e.g. a mixer collecting funds) vs. many outputs
(dispersal, e.g. layering funds out to many wallets). Splitting them is
strictly more informative and costs nothing extra to compute.

Betweenness centrality was dropped after an empirical stability check
(caught in review, verified directly rather than assumed either way):
computing it twice with different seeds at this project's k=500 gave
Pearson r=0.70 but Spearman rho=0.33 between the two runs, and only
~4% of nodes had a nonzero value at all. A rank correlation that weak
between two runs of the "same" measurement means the feature is mostly
sampling noise, not signal - a tree model would be learning from which
500 nodes happened to get sampled, not from real structural information.
Raising k enough to fix this (multiple thousand) was tested and found
impractical within reasonable runtime on this graph size; dropping the
feature is the honest choice given that constraint, not silently keeping
a noisy feature and hoping it nets out to nothing.
"""
import pickle
import networkx as nx

with open('data/graph.pkl', 'rb') as f:
    G = pickle.load(f)

print("Computing in-degree and out-degree centrality...")
in_degree_cent = nx.in_degree_centrality(G)
out_degree_cent = nx.out_degree_centrality(G)

with open('data/centrality_data.pkl', 'wb') as f:
    pickle.dump({'in_degree': in_degree_cent, 'out_degree': out_degree_cent}, f)

print("Centrality data saved (in/out degree only - see module docstring for why "
      "betweenness was dropped).")
