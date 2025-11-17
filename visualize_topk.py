# visualize_topk.py
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import data_gen, features
import numpy as np

# Rebuild graph the same way used for model training to preserve node indices
G, labels_df, inj = data_gen.build_synthetic_graph(n_accounts=1000, n_cycles=6, cycle_size=4, n_rings=3, ring_size=6)
feats = features.get_basic_node_features(G)
cycle_feats = features.compute_cycle_features(G, max_cycle_length=6)
G = features.attach_features_to_graph(G, feats, cycle_feats)

# Load scores
scores = pd.read_csv("node_scores.csv")
topk = scores.head(30)['node_index'].tolist()

# Build induced subgraph of topk + their immediate neighbors (pred and succ)
nodes_to_plot = set(topk)
for n in topk:
    nodes_to_plot.update(list(G.successors(int(n)))[:10])
    nodes_to_plot.update(list(G.predecessors(int(n)))[:10])

sub = G.subgraph(nodes_to_plot).copy()
pos = nx.spring_layout(sub, seed=42)
plt.figure(figsize=(12, 8))
node_colors = ['red' if sub.nodes[n].get('label',0)==1 else 'green' for n in sub.nodes()]
nx.draw_networkx_nodes(sub, pos, node_size=200, node_color=node_colors)
nx.draw_networkx_edges(sub, pos, arrows=True, alpha=0.6)
nx.draw_networkx_labels(sub, pos, font_size=8)
plt.title("Top suspicious nodes and neighbors (red = suspicious label)")
plt.axis('off')
plt.tight_layout()
plt.savefig("topk_subgraph.png", dpi=200)
print("Saved topk_subgraph.png (open this file to view)")
