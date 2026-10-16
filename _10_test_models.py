import pandas as pd
import numpy as np
import re
import ast
import os
cma = __import__('custom-model-assessment')
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv

NODE_DIM = 49  # 4 one-hot + angle + torsion features
EDGE_DIM = 2   # distance value + is_consecutive flag


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions (copied from original)
# ──────────────────────────────────────────────────────────────────────────────

def get_all_indexes_from_string(string):
    """string might contain any number of single digit numbers following one
    another, e.g ts125 — returns [1, 2, 5]."""
    return [int(char) for char in string if char.isdigit()]


def get_graph_hot_encoding_v3(row, cols):
    NUM_NODES = 6  # we only care about nucleotides 1-6
    d = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    num_classes = len(d)

    seq = row['seq'][:NUM_NODES]  # slice to first 6 nucleotides
    node_features = torch.nn.functional.one_hot(
        torch.tensor([d[n] for n in seq], dtype=torch.long),
        num_classes=num_classes
    ).to(torch.float32)

    edge_dict    = {}
    node_angle   = {i: [] for i in range(NUM_NODES)}
    node_torsion = {i: [] for i in range(NUM_NODES)}

    feature_cols = [c for c in cols if c not in ('seq', 'is_positive', 'source_file')]

    for col in feature_cols:
        idxs = get_all_indexes_from_string(col)
        val  = float(row[col]) if not pd.isna(row[col]) else 0.0

        if any(i > NUM_NODES for i in idxs):
            continue

        if len(idxs) == 2:
            i, j = idxs[0] - 1, idxs[1] - 1
            edge_dict.setdefault((i, j), []).append(val)

        elif len(idxs) == 3:
            middle = idxs[1] - 1
            node_angle[middle].append(val)

        elif len(idxs) == 4:
            mid1, mid2 = idxs[1] - 1, idxs[2] - 1
            node_torsion[mid1].append(val)
            node_torsion[mid2].append(val)

    max_angle   = max((len(v) for v in node_angle.values()),   default=0)
    max_torsion = max((len(v) for v in node_torsion.values()), default=0)

    extra_node_feats = []
    for i in range(NUM_NODES):
        a_feats = node_angle[i]   + [0.0] * (max_angle   - len(node_angle[i]))
        t_feats = node_torsion[i] + [0.0] * (max_torsion - len(node_torsion[i]))
        extra_node_feats.append(a_feats + t_feats)

    extra = torch.tensor(extra_node_feats, dtype=torch.float32)
    x = torch.cat([node_features, extra], dim=1)

    edge_index_list = []
    edge_attr_list  = []

    for (i, j), weights in edge_dict.items():
        is_consecutive = 1.0 if abs(i - j) == 1 else 0.0
        edge_attr_list.append(weights + [is_consecutive])
        edge_index_list.append([i, j])

    edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float32)
    edge_index = torch.tensor(edge_index_list, dtype=torch.int64).t().contiguous()

    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_attr  = torch.cat([edge_attr,  edge_attr],          dim=0)

    y = torch.tensor([int(row['is_positive'])], dtype=torch.int64)
    return Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y)


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (copied from original)
# ──────────────────────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GATv2Conv(NODE_DIM, 32, edge_dim=EDGE_DIM, heads=1)
        self.conv2 = GATv2Conv(32, 64, edge_dim=EDGE_DIM, heads=1)
        self.conv3 = GATv2Conv(64, 32, edge_dim=EDGE_DIM, heads=1)
        self.lin   = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.lin(x)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (copied from original)
# ──────────────────────────────────────────────────────────────────────────────

def test(model, loader, return_predictions=False):
    model.eval()
    correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())

            if return_predictions:
                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(data.y.cpu().numpy().tolist())

    accuracy = correct / len(loader.dataset)
    if return_predictions:
        return accuracy, np.array(all_preds), np.array(all_labels)
    return accuracy


def test_proba(model, loader):
    """Returns softmax probabilities instead of hard predictions."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Build the final-eval loader (copied from original)
# ──────────────────────────────────────────────────────────────────────────────

dftofinalevaltmp = pd.read_csv("filtered_geometric_features_to_test8a.csv", sep=',', index_col=0)
dftofinaleval_X  = dftofinalevaltmp.drop(columns=['is_positive'])
dftofinaleval_Y  = dftofinalevaltmp['is_positive']

graph_dftofinaleval = dftofinalevaltmp.apply(
    lambda x: get_graph_hot_encoding_v3(x, dftofinalevaltmp.columns[:-1]), axis=1
)
final_eval_loader = DataLoader(graph_dftofinaleval, batch_size=32)


# ──────────────────────────────────────────────────────────────────────────────
# Load every .pth model from "models_that_worked" and evaluate
# ──────────────────────────────────────────────────────────────────────────────

models_dir = "test_models_annealing"
#models_dir = "models_that_worked"

if not os.path.isdir(models_dir):
    raise FileNotFoundError(f"Directory '{models_dir}' not found. "
                            "Please create it and place your .pth files inside.")

model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pth")])

if not model_files:
    raise FileNotFoundError(f"No .pth files found in '{models_dir}'.")

print(f"Found {len(model_files)} model(s) in '{models_dir}':\n")

all_results = []

for model_filename in model_files:
    model_path = os.path.join(models_dir, model_filename)
    print(f"{'='*60}")
    print(f"Evaluating: {model_filename}")
    print(f"{'='*60}")

    # Instantiate a fresh model and load weights
    model = GCN(hidden_channels=64)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Hard predictions
    acc, preds, labels = test(model, final_eval_loader, return_predictions=True)
    f1  = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    # Soft probabilities (class 1)
    probs, _ = test_proba(model, final_eval_loader)
    cmascore =cma.customAssesment2(labels, preds)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"  Real labels      : {dftofinaleval_Y.values}")
    print(f"  Model labels     : {labels}")
    print(f"  Model predictions: {preds}")
    print(f"  Probabilities (class 1): {np.round(probs, 3)}")
    print(f'  Custom Assessment Score: {cmascore}')
    print()

    all_results.append({
        'model': model_filename,
        'accuracy': acc,
        'f1': f1,
        'mcc': mcc,
        'cmascore': cmascore
    })

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<40} {'Acc':>8} {'F1':>8} {'MCC':>8} {'CMA':>8}")
print("-" * 60)
for r in all_results:
    print(f"{r['model']:<40} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['mcc']:>8.4f} {r['cmascore']:>8}")
print(f"{'='*60}")