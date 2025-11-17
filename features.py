# features.py
import networkx as nx
import numpy as np
from collections import deque

def get_basic_node_features(G):
    """
    Compute a set of per-node features using NetworkX for a DiGraph G.
    Returns: DataFrame-like dict keyed by node id -> feature vector dict
    Features:
      - in_degree, out_degree
      - weighted_in_degree, weighted_out_degree (by amount)
      - total_amount_in, total_amount_out
      - avg_edge_amount_out
      - pagerank, closeness, betweenness (approx)
      - clustering coefficient (on undirected version)
    """
    nodes = list(G.nodes())
    feats = {}

    # degrees
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    # Weighted sums:
    amount_in = {n: 0.0 for n in nodes}
    amount_out = {n: 0.0 for n in nodes}
    cnt_in = {n: 0 for n in nodes}
    cnt_out = {n: 0 for n in nodes}
    for u, v, d in G.edges(data=True):
        a = float(d.get('amount', 0.0))
        c = int(d.get('cnt', 1))
        amount_out[u] += a
        cnt_out[u] += c
        amount_in[v] += a
        cnt_in[v] += c

    # centrality measures
    try:
        pagerank = nx.pagerank(G.to_undirected(), alpha=0.85)
    except Exception:
        pagerank = {n: 0.0 for n in nodes}
    try:
        closeness = nx.closeness_centrality(G)
    except Exception:
        closeness = {n: 0.0 for n in nodes}
    # approximate betweenness to save time
    betweenness = nx.betweenness_centrality(G, k=min(100, max(10, int(len(nodes) * 0.02))))
    clustering = nx.clustering(G.to_undirected())

    for n in nodes:
        avg_out_amt = amount_out[n] / out_deg.get(n, 1) if out_deg.get(n, 0) > 0 else 0.0
        feats[n] = {
            'in_degree': float(in_deg.get(n, 0)),
            'out_degree': float(out_deg.get(n, 0)),
            'total_amount_in': float(amount_in[n]),
            'total_amount_out': float(amount_out[n]),
            'cnt_in': float(cnt_in[n]),
            'cnt_out': float(cnt_out[n]),
            'avg_out_amount': float(avg_out_amt),
            'pagerank': float(pagerank.get(n, 0.0)),
            'closeness': float(closeness.get(n, 0.0)),
            'betweenness': float(betweenness.get(n, 0.0)),
            'clustering': float(clustering.get(n, 0.0)),
        }
    return feats

def find_directed_cycles_limited(G, max_cycle_length=6):
    """
    Find simple directed cycles up to max_cycle_length.
    Returns dict node -> number of cycles it participates in.
    Warning: can be expensive; we limit lengths to small values.
    """
    cycles_count = {n: 0 for n in G.nodes()}
    # Use a bounded DFS starting from each node (tuned for small cycles)
    for start in G.nodes():
        stack = [(start, [start])]
        visited = set()
        while stack:
            node, path = stack.pop()
            if len(path) > max_cycle_length:
                continue
            for nbr in G.successors(node):
                if nbr == start and len(path) >= 2:
                    # found cycle
                    for p in path:
                        cycles_count[p] += 1
                elif nbr not in path and nbr > start:  # small pruning to reduce duplicates
                    stack.append((nbr, path + [nbr]))
    return cycles_count

def compute_cycle_features(G, max_cycle_length=6):
    cycles_participation = find_directed_cycles_limited(G, max_cycle_length)
    # normalize
    maxc = max(cycles_participation.values()) if cycles_participation else 1
    return {n: cycles_participation[n] / max(1, maxc) for n in cycles_participation}

def detect_ring_like_subgraphs(G, min_size=4, density_threshold=0.5):
    """
    Heuristic: find subgraphs with high internal transfer density and high mean amounts.
    Use connected components of undirected version as candidates.
    Returns set of node ids flagged as ring-like.
    """
    U = G.to_undirected()
    ring_nodes = set()
    for comp in nx.connected_components(U):
        if len(comp) < min_size:
            continue
        sub = G.subgraph(comp)
        # compute directed-edge density: existing edges / max possible directed edges
        n = len(sub)
        possible = n*(n-1)
        ecount = sub.number_of_edges()
        density = ecount / possible if possible > 0 else 0.0
        # mean internal amount
        amounts = [d.get('amount', 0.0) for _,_,d in sub.edges(data=True)]
        mean_amount = sum(amounts)/len(amounts) if amounts else 0.0
        if density >= density_threshold and mean_amount > 5000:
            ring_nodes.update(comp)
    return ring_nodes

def attach_features_to_graph(G, feats_dict, cycle_feats):
    """
    Attach computed feature fields to G.nodes[].feature_vector (list)
    """
    for n in G.nodes():
        base = feats_dict.get(n, {})
        cf = cycle_feats.get(n, 0.0)
        # simple feature vector
        vec = [
            base.get('in_degree', 0.0),
            base.get('out_degree', 0.0),
            base.get('total_amount_in', 0.0),
            base.get('total_amount_out', 0.0),
            base.get('cnt_in', 0.0),
            base.get('cnt_out', 0.0),
            base.get('avg_out_amount', 0.0),
            base.get('pagerank', 0.0),
            base.get('closeness', 0.0),
            base.get('betweenness', 0.0),
            base.get('clustering', 0.0),
            cf
        ]
        G.nodes[n]['feat_vec'] = np.array(vec, dtype=float)
    return G
