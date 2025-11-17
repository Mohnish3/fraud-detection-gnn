# Financial Fraud Detection using Graph Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-%23007ACC.svg?logo=networkx&logoColor=white)](https://networkx.org/)

A comprehensive framework for detecting fraudulent financial activities using Graph Neural Networks (GNNs). This project demonstrates how graph-based approaches can effectively identify suspicious transaction patterns like circular transactions and money laundering rings.

## 📑 Table of Contents

- [Financial Fraud Detection using Graph Neural Networks](#financial-fraud-detection-using-graph-neural-networks)
  - [Key Features](#-key-features)
  - [Performance Metrics](#-performance-metrics)
  - [System Architecture](#️-system-architecture)
    - [Core Components](#core-components)

    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Basic Usage](#basic-usage)
  - [Project Structure](#-project-structure)
  - [Advanced Configuration](#-advanced-configuration)
    - [Training Parameters](#training-parameters)
    - [Key Arguments](#key-arguments)
  - [Model Architecture](#-model-architecture)
  - [Fraud Patterns Detected](#-fraud-patterns-detected)
    - [1. Circular Transactions](#1-circular-transactions)
    - [2. Money Laundering Rings](#2-money-laundering-rings)
  - [Generated Outputs](#-generated-outputs)
  - [Example Results](#-example-results)
  - [Methodology](#-methodology)
    - [Feature Engineering](#feature-engineering)
    - [Data Splitting](#data-splitting)
  - [Development](#️-development)
    - [Adding New Features](#adding-new-features)
    - [Custom Fraud Patterns](#custom-fraud-patterns)
 

## Key Features

- **Synthetic Data Generation**: Privacy-preserving synthetic financial transaction graphs
- **Graph Feature Engineering**: Comprehensive node features including centrality measures and cycle participation
- **GNN Model**: 2-layer Graph Convolutional Network for node classification
- **Interpretable Results**: Visualization of suspicious subgraphs and model performance
- **Production-Ready Pipeline**: End-to-end workflow from data generation to visualization

## Performance Metrics

| Metric | Value |
|--------|-------|
| Validation AUC | 0.91 |
| Test AUC | 0.89 |
| Accuracy | 0.93 |
| Precision | 0.88 |
| Recall | 0.85 |
| F1-Score | 0.86 |

## System Architecture
Data Generation → Feature Extraction → Dataset Preparation → GNN Training → Evaluation → Visualization


### Core Components

- **`data_gen.py`**: Synthetic graph generation with injected fraud patterns
- **`features.py`**: Graph feature computation (centrality, cycles, etc.)
- **`dataset.py`**: PyTorch Geometric data conversion
- **`model.py`**: GCN model architecture
- **`train_improved.py`**: Training pipeline with early stopping
- **`visualize_topk.py`**: Suspicious nodes visualization


### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohnish3/fraud-detection-gnn.git
   cd fraud-graph-gnn

2. Set up virtual environment (Windows)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

3. Alternative: Use setup script (Windows)
   .\setup_windows.ps1


# Basic Usage

1. **Train the model**
   python train_improved.py --n_nodes 1000 --epochs 80
2. Generate visualization
   python visualize_topk.py
3. View Results
   Start-Process .\topk_subgraph.png

# Project Structure
fraud-graph-gnn/ \
├── data_gen.py           # Synthetic graph generation\
├── features.py           # Graph feature extraction\
├── dataset.py            # PyTorch Geometric data conversion\
├── model.py              # GCN model architecture\
├── train_improved.py     # Enhanced training pipeline\
├── visualize_topk.py     # Suspicious nodes visualization\
├── utils.py              # Utility functions\
├── requirements.txt      # Python dependencies\
├── setup_windows.ps1     # Windows setup script\
└── README.md


# Advanced Confirguration

## Training Parameters

### Custom training configuration
python train_improved.py \
  --n_nodes 2000 \
  --epochs 100 \
  --hidden 128 \
  --dropout 0.3 \
  --lr 0.001

### Key Training Arguments

--n_nodes: Number of accounts in synthetic graph (default: 1000) \
--epochs: Maximum training epochs (default: 100)\
--hidden: Hidden layer dimension (default: 64)\
--dropout: Dropout rate (default: 0.5)\
--lr: Learning rate (default: 0.01)

# Model Architecture

Architecture: 2-layer Graph Convolutional Network (GCN)
Input Features: 12-dimensional node features
Hidden Dimension: 64 units
Activation: ReLU
Dropout: 0.5
Loss Function: Weighted Cross Entropy
Optimizer: Adam (lr=0.01, weight_decay=1e-4)

# Fraud Patterns Detected

## 1. Circular Transactions

Multiple accounts transferring money in circular paths
High-frequency, structured payment cycles
Detected through cycle participation features

## 2. Money Laundering Rings

Densely connected subgraphs
High internal transaction density
Structured to obscure money origins

# Generated Outputs

best_model_improved.pt: Trained model weights
node_scores.csv: Fraud probability scores for all nodes
train_history.csv: Training metrics history
topk_subgraph.png: Visualization of suspicious subgraph
injected_nodes.json: Ground truth injection patterns

# Example Results

<img width="701" height="579" alt="Picture2" src="https://github.com/user-attachments/assets/93a5576b-1290-42b2-aedf-42a6fe3172e6" />

*ROC Curve showing model*

<img width="1430" height="953" alt="Picture1" src="https://github.com/user-attachments/assets/519ec0d5-8cde-455f-a65e-314dccdaae47" />

*Top suspicious nodes and neighbors (red = fraudulent)*

# Methodology

## Feature Engineering

Structural Features: In/Out degree, clustering coefficient
Centrality Measures: PageRank, Closeness, Betweenness
Transaction Features: Amount statistics, frequency
Behavioral Features: Cycle participation scores

## Data Splitting

Training: 70% of nodes
Validation: 15% of nodes
Test: 15% of nodes

# Development

## Adding New Features

Extend features.py with new feature computation
Update feature dimension in model.py
Retrain model with updated features

## Custom Fraud Patterns

Modify data_gen.py to inject new pattern types
Update feature computation if needed
Retrain and evaluate model performance

   
