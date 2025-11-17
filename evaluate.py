# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import nx_to_pyg_data
from model import GCNNodeClassifier
from sklearn.metrics import roc_curve, auc
import data_gen, features

def load_data_and_model(model_path="best_model.pt", device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    G, labels_df, inj = data_gen.build_synthetic_graph(n_accounts=1000, n_cycles=6, cycle_size=4, n_rings=3, ring_size=6)
    feats = features.get_basic_node_features(G)
    cycle_feats = features.compute_cycle_features(G, max_cycle_length=6)
    G = features.attach_features_to_graph(G, feats, cycle_feats)
    data = nx_to_pyg_data(G)
    model = GCNNodeClassifier(in_dim=data.x.shape[1], hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return G, data, model, device, inj

def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], '--', label='random')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    plt.show()

if __name__ == "__main__":
    G, data, model, device, inj = load_data_and_model()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[:,1]
    y = data.y.numpy()
    # simple ROC on all nodes
    plot_roc(y, probs)
    # print top predicted suspicious accounts
    topk = np.argsort(-probs)[:30]
    print("Top predicted suspicious node indices and scores:")
    for idx in topk[:20]:
        print(idx, f"{probs[idx]:.3f}", "label=", int(y[idx]))
