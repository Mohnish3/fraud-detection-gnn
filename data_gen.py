# data_gen.py
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

RANDOM_SEED = 42

def set_seed(s=RANDOM_SEED):
    random.seed(s)
    np.random.seed(s)

def generate_base_graph(n_accounts=1000, avg_degree=3, p_heavy=0.05):
    """
    Generate a base directed transaction graph using configuration model-like sampling.
    Returns a NetworkX DiGraph with edge attributes: amount, cnt (frequency).
    """
    set_seed()
    G = nx.DiGraph()
    for i in range(n_accounts):
        G.add_node(i, account_id=i)

    # For each node, create Poisson(avg_degree) outgoing edges
    for u in G.nodes():
        k = np.random.poisson(avg_degree)
        for _ in range(k):
            v = np.random.randint(0, n_accounts)
            if u == v:
                continue
            amount = float(np.round(np.random.exponential(5000.0), 2))  # typical transfer amount
            if random.random() < p_heavy:
                # occasional large transfer
                amount *= np.random.uniform(10, 50)
            cnt = np.random.randint(1, 5)
            if G.has_edge(u, v):
                G[u][v]['amount'] += amount
                G[u][v]['cnt'] += cnt
            else:
                G.add_edge(u, v, amount=amount, cnt=cnt)
    return G

def inject_circular_scheme(G, cycle_size=4, amount_scale=10000.0):
    """
    Inject a directed cycle (circular transactions) among cycle_size new or existing nodes.
    Mark nodes as part of circular pattern with 'label_source' attribute.
    """
    nodes = list(G.nodes())
    # pick nodes (try to pick distinct nodes)
    if len(nodes) < cycle_size:
        raise ValueError("Graph too small for cycle injection")

    chosen = random.sample(nodes, cycle_size)
    # add edges forming a directed cycle with repeated transactions
    for i in range(cycle_size):
        u = chosen[i]
        v = chosen[(i+1) % cycle_size]
        amount = np.random.uniform(amount_scale * 0.8, amount_scale * 1.2)
        cnt = np.random.randint(5, 15)  # high frequency
        if G.has_edge(u, v):
            G[u][v]['amount'] += amount
            G[u][v]['cnt'] += cnt
        else:
            G.add_edge(u, v, amount=amount, cnt=cnt)
        # mark nodes as circular candidates
        G.nodes[u].setdefault('injected', []).append('circular')
    return chosen

def inject_laundering_ring(G, ring_size=6, internal_density=0.8, amount_scale=20000.0):
    """
    Create a dense subgraph (directed) among ring_size nodes where lots of internal transfers occur.
    This models a money-laundering ring.
    Returns chosen node list.
    """
    nodes = list(G.nodes())
    chosen = random.sample(nodes, ring_size)
    for u in chosen:
        for v in chosen:
            if u == v:
                continue
            if random.random() < internal_density:
                amount = np.random.uniform(amount_scale * 0.5, amount_scale * 1.5)
                cnt = np.random.randint(3, 20)
                if G.has_edge(u, v):
                    G[u][v]['amount'] += amount
                    G[u][v]['cnt'] += cnt
                else:
                    G.add_edge(u, v, amount=amount, cnt=cnt)
    for u in chosen:
        G.nodes[u].setdefault('injected', []).append('laundering_ring')
    return chosen

def label_nodes_from_injections(G):
    """
    Create a ground-truth 'label' node attribute:
     - 1 suspicious if node has injected patterns
     - 0 otherwise
    Also produce a pandas DataFrame per-node labels for convenience.
    """
    labels = {}
    for n in G.nodes():
        tags = G.nodes[n].get('injected', [])
        labels[n] = 1 if tags else 0
        G.nodes[n]['label'] = labels[n]
    return pd.DataFrame.from_dict(labels, orient='index', columns=['label'])

def build_synthetic_graph(n_accounts=1000,
                          n_cycles=5, cycle_size=4,
                          n_rings=3, ring_size=6):
    """
    Generates a synthetic graph and injects specified number of circular transactions and laundering rings.
    Returns: G (nx.DiGraph), labels_df (pandas.DataFrame)
    """
    set_seed()
    G = generate_base_graph(n_accounts=n_accounts)
    injected_nodes = {'circular': [], 'ring': []}
    for _ in range(n_cycles):
        chosen = inject_circular_scheme(G, cycle_size=cycle_size)
        injected_nodes['circular'].extend(chosen)
    for _ in range(n_rings):
        chosen = inject_laundering_ring(G, ring_size=ring_size)
        injected_nodes['ring'].extend(chosen)
    labels_df = label_nodes_from_injections(G)
    return G, labels_df, injected_nodes

if __name__ == "__main__":
    G, labels_df, inj = build_synthetic_graph(n_accounts=1000, n_cycles=8, cycle_size=4,
                                              n_rings=4, ring_size=8)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    print("Suspicious nodes (label=1):", labels_df['label'].sum())
    labels_df.to_csv("node_labels.csv")
