# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import NeighborSampler
from dataset import nx_to_pyg_data
from model import GCNNodeClassifier
from utils import set_seed, compute_metrics
import argparse
import numpy as np

def train_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None)
    loss = F.cross_entropy(out, data.y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y = data.y.cpu().numpy()
    return probs, preds, y

def main(args):
    set_seed(args.seed)
    import data_gen, features, dataset
    # build data
    G, labels_df, inj = data_gen.build_synthetic_graph(n_accounts=args.n_nodes,
                                                       n_cycles=args.n_cycles, cycle_size=args.cycle_size,
                                                       n_rings=args.n_rings, ring_size=args.ring_size)
    feats = features.get_basic_node_features(G)
    cycle_feats = features.compute_cycle_features(G, max_cycle_length=6)
    G = features.attach_features_to_graph(G, feats, cycle_feats)
    data = dataset.nx_to_pyg_data(G, normalize_features=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    model = GCNNodeClassifier(in_dim=data.x.shape[1], hidden_dim=args.hidden, num_layers=2, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple train/val/test split on nodes: use indices
    n = data.num_nodes
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:int(0.7*n)]
    val_idx = idx[int(0.7*n):int(0.85*n)]
    test_idx = idx[int(0.85*n):]

    # Create masks on data for ease
    mask = torch.zeros(n, dtype=torch.bool)
    train_mask = mask.clone()
    val_mask = mask.clone()
    test_mask = mask.clone()
    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True
    test_mask[torch.tensor(test_idx)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    best_val_auc = 0.0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, data, optimizer, device)
        probs, preds, y = evaluate(model, data, device)
        # compute metrics on val set
        val_probs = probs[data.val_mask.cpu().numpy()]
        val_y = y[data.val_mask.cpu().numpy()]
        from utils import compute_metrics
        val_report, val_auc = compute_metrics(val_y, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), args.model_path)
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:03d} Loss: {loss:.4f} Val AUC: {val_auc:.4f} (best {best_val_auc:.4f})")
    # final test
    model.load_state_dict(torch.load(args.model_path))
    probs, preds, y = evaluate(model, data, device)
    test_probs = probs[data.test_mask.cpu().numpy()]
    test_y = y[data.test_mask.cpu().numpy()]
    test_report, test_auc = compute_metrics(test_y, test_probs)
    print("Test AUC:", test_auc)
    print("Classification report (test):")
    import json
    print(json.dumps(test_report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--n_cycles", type=int, default=6)
    parser.add_argument("--cycle_size", type=int, default=4)
    parser.add_argument("--n_rings", type=int, default=3)
    parser.add_argument("--ring_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--force_cpu", action='store_true')
    args = parser.parse_args()
    main(args)
