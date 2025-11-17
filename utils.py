# utils.py
import random
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    y_true: np array
    y_pred_probs: N x 2 softmax probabilities or single column prob for class 1
    """
    if y_pred_probs.ndim == 2:
        probs = y_pred_probs[:,1]
    else:
        probs = y_pred_probs
    y_hat = (probs >= threshold).astype(int)
    report = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    return report, auc
