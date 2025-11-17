# train_improved.py
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from utils import set_seed, compute_metrics
from model import GCNNodeClassifier
import data_gen, features, dataset
import argparse
import pandas as pd

def train_epoch(model, data, optimizer, device, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None)
    loss = loss_fn(out, data.y.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, device, mask=None):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device) if data.edge_attr is not None else None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y = data.y.cpu().numpy()
    if mask is None:
        return probs, preds, y
    mask_idx = mask.cpu().numpy()
    return probs[mask_idx], preds[mask_idx], y[mask_idx]

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def main(args):
    set_seed(args.seed)
    # Build synthetic dataset (same pipeline as before)
    G, labels_df, inj = data_gen.build_synthetic_graph(n_accounts=args.n_nodes,
                                                       n_cycles=args.n_cycles, cycle_size=args.cycle_size,
                                                       n_rings=args.n_rings, ring_size=args.ring_size)
    feats = features.get_basic_node_features(G)
    cycle_feats = features.compute_cycle_features(G, max_cycle_length=6)
    G = features.attach_features_to_graph(G, feats, cycle_feats)
    data = dataset.nx_to_pyg_data(G, normalize_features=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    model = GCNNodeClassifier(in_dim=data.x.shape[1], hidden_dim=args.hidden, num_layers=2, dropout=args.dropout).to(device)

    # Compute class weights to handle imbalance
    labels = data.y.numpy()
    unique, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(unique, counts))
    # avoid zero division
    total = labels.shape[0]
    class_weights = []
    for c in range(2):
        class_weights.append(total / (1 + freq.get(c, 0)))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights.cpu().numpy())

    loss_fn = lambda logits, target: F.cross_entropy(logits, target, weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Split indices
    n = data.num_nodes
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:int(0.7*n)]
    val_idx = idx[int(0.7*n):int(0.85*n)]
    test_idx = idx[int(0.85*n):]

    # masks
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
    epochs_no_improve = 0
    history = []
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        loss = train_epoch(model, data, optimizer, device, loss_fn)
        val_probs, _, val_y = evaluate(model, data, device, mask=data.val_mask)
        val_report, val_auc = compute_metrics(val_y, val_probs)
        history.append({'epoch': epoch, 'loss': loss, 'val_auc': val_auc})
        # early stopping check
        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            epochs_no_improve = 0
            save_checkpoint(model, args.model_path)
        else:
            epochs_no_improve += 1

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} Loss: {loss:.4f} Val AUC: {val_auc:.4f} (best {best_val_auc:.4f}) time:{time.time()-t0:.2f}s")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break

    # Save training history
    pd.DataFrame(history).to_csv(args.history_path, index=False)
    print("Training finished. Best val AUC:", best_val_auc)
    # Load best model and perform test eval
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_probs, test_preds, test_y = evaluate(model, data, device, mask=data.test_mask)
    test_report, test_auc = compute_metrics(test_y, test_probs)
    print("Test AUC:", test_auc)
    print(json.dumps(test_report, indent=2))
    # Save final predictions for all nodes
    with torch.no_grad():
        logits_all = model(data.x.to(device), data.edge_index.to(device))
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()[:,1]
    df = pd.DataFrame({'node_index': np.arange(len(probs_all)), 'score': probs_all, 'label': data.y.numpy()})
    df.to_csv(args.scores_path, index=False)
    print("Saved node scores to", args.scores_path)
    # Save injected ground truth info to inspect (if data_gen returned it)
    try:
        with open(args.injected_path, 'w') as f:
            json.dump(inj, f)
        print("Saved injected patterns info to", args.injected_path)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1000)
    parser.add_argument("--n_cycles", type=int, default=6)
    parser.add_argument("--cycle_size", type=int, default=4)
    parser.add_argument("--n_rings", type=int, default=3)
    parser.add_argument("--ring_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="best_model_improved.pt")
    parser.add_argument("--history_path", type=str, default="train_history.csv")
    parser.add_argument("--scores_path", type=str, default="node_scores.csv")
    parser.add_argument("--injected_path", type=str, default="injected_nodes.json")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--force_cpu", action='store_true')
    args = parser.parse_args()
    main(args)
