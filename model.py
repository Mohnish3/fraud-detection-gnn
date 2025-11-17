# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCNNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.mlp(x)
        return out

# Alternate model (GAT) if you'd like:
class GATNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=4, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.mlp(x)
        return x
