# save_scores.py
import torch, numpy as np, pandas as pd
from model import GCNNodeClassifier
from dataset import nx_to_pyg_data
import data_gen, features

G, labels_df, inj = data_gen.build_synthetic_graph(n_accounts=1000, n_cycles=6, cycle_size=4, n_rings=3, ring_size=6)
feats = features.get_basic_node_features(G)
cycle_feats = features.compute_cycle_features(G, max_cycle_length=6)
G = features.attach_features_to_graph(G, feats, cycle_feats)
data = nx_to_pyg_data(G)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNNodeClassifier(in_dim=data.x.shape[1], hidden_dim=64)
model.load_state_dict(torch.load("best_model_improved.pt", map_location=device))
model.to(device).eval()

with torch.no_grad():
    logits = model(data.x.to(device), data.edge_index.to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[:,1]

df = pd.DataFrame({
    'node_index': np.arange(len(probs)),
    'score': probs,
    'label': data.y.numpy()
})
df = df.sort_values('score', ascending=False)
df.to_csv("node_scores.csv", index=False)
print("Saved node_scores.csv; top 10:")
print(df.head(10))
