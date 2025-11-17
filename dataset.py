# dataset.py
import torch
from torch_geometric.data import Data
import numpy as np

def nx_to_pyg_data(G, normalize_features=True):
    """
    Convert directed NetworkX DiGraph into a torch_geometric.data.Data object.
    Node features: G.nodes[n]['feat_vec']
    Edge attributes: amount, cnt -> store as edge_attr = [amount, cnt]
    Labels: G.nodes[n]['label'] (0/1)
    """
    node_list = list(G.nodes())
    node_to_idx = {n:i for i,n in enumerate(node_list)}
    n_nodes = len(node_list)

    # Build edge_index
    edge_index = [[], []]
    edge_attrs = []

    for u, v, d in G.edges(data=True):
        edge_index[0].append(node_to_idx[u])
        edge_index[1].append(node_to_idx[v])
        amount = float(d.get('amount', 0.0))
        cnt = float(d.get('cnt', 0.0))
        edge_attrs.append([amount, cnt])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0,2), dtype=torch.float)

    # Node features
    feats = []
    labels = []
    for n in node_list:
        fv = G.nodes[n].get('feat_vec')
        if fv is None:
            # fallback to zeros
            fv = np.zeros(12, dtype=float)
        feats.append(fv)
        labels.append(int(G.nodes[n].get('label', 0)))

    x = torch.tensor(np.vstack(feats), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Optionally normalize features (column-wise)
    if normalize_features:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-9
        x = (x - mean) / std

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data
