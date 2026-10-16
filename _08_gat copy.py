import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
import csv
from itertools import combinations
import json, os
from imblearn.under_sampling import CondensedNearestNeighbour

cma = __import__('custom-model-assessment')
filter_script = __import__('08b-filter-pdb-by-date')

from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.decomposition import PCA


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from sklearn.decomposition import PCA

from torch.nn import Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import copy
from torch_geometric.nn import GATv2Conv
import random

NODE_DIM = 49  # 4 one-hot + angle + torsion features
EDGE_DIM = 2   # distance value + is_consecutive flag
SEED = 42
FOLDER = "test_models_annealing"
SCORE_THRESHOLD = -9


def setSeed(seed, folder, scorethr):
    global SEED, FOLDER, SCORE_THRESHOLD
    SEED = seed
    FOLDER = folder
    SCORE_THRESHOLD = scorethr
    main()

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_sequences(seqs, path, max_len):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(seqs) >= max_len:
                break
            seq = row[0].strip()
            # Only keep rows with uppercase letters (A-Z), ignore headers like "sequence"
            if re.fullmatch(r"[A-Z]+", seq):
                seqs.append(seq)


def parse_point(cell):
    if pd.isna(cell):
        return cell

    if isinstance(cell, np.ndarray):
        return cell.flatten()

    if isinstance(cell, list):
        return np.array(cell, dtype=float).flatten()

    if isinstance(cell, str):
        s = cell.strip()

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                arr = np.array(parsed, dtype=float)
                return arr.flatten() if arr.ndim > 1 else arr
        except Exception:
            pass

        try:
            nums = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', s)]
            if len(nums) == 3:
                return np.array(nums, dtype=float)
            elif len(nums) == 6:
                arr = np.array(nums, dtype=float).reshape(2, 3)
                return arr[0]
            else:
                return cell
        except Exception:
            return cell

    return cell


def count_planar_angle(p1, p2, p3):
    print(f'counting planar angle for {p1}  {p2}   {p3}')
    b1 = p2 - p1
    b2 = p2 - p3
    angle = np.arccos(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2)))
    return np.degrees(angle)


def count_torsion_angle(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    torsion = np.arctan2(
        np.dot(np.cross(n1, n2), b2 * np.linalg.norm(b2)), np.dot(n1, n2)
    )
    return np.degrees(torsion)


def count_euclid_dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def count_planar_angle_piel(p1, p2, p3):
    for idx, p in enumerate([p1, p2, p3], start=1):
        print(f"p{idx}: type={type(p)}, value={p}")
    return np.dot(p2 - p1, p3 - p1)


def get_graph(row, cols):
    edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [],
                  (1,2): [], (1,3): [], (1,4): [], (1,5): [],
                  (2,3): [], (2,4): [], (2,5): [],
                  (3,4): [], (3,5): [],
                  (4,5): []}

    for i in range(len(cols)):
        weigth = row[i]
        if len(cols[i]) == 2:
            edges_dict[(cols[i][0], cols[i][1])].append(weigth)

    edge_attr = []
    for k, v in edges_dict.items():
        edge_attr.append(v)
        print(f'{k},{len(v)}')
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    edge_index_list = torch.tensor([edge for edge in edges_dict.keys()], dtype=torch.int64).t().contiguous()
    edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)

    d = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
    y = torch.tensor([row['class']], dtype=torch.int64)
    x = torch.tensor([[d[n]] for n in row['seq']], dtype=torch.float32)
    graph = Data(edge_index=edge_index_list, edge_attr=edge_attr, y=y, x=x)
    return graph


def get_graph_hot_encoding(row, cols):
    edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [],
                  (1,2): [], (1,3): [], (1,4): [], (1,5): [],
                  (2,3): [], (2,4): [], (2,5): [],
                  (3,4): [], (3,5): [],
                  (4,5): []}

    for i in range(len(cols)):
        weigth = row[i]
        if len(cols[i]) == 2:
            edges_dict[(cols[i][0], cols[i][1])].append(weigth)

    edge_attr = []
    for k, v in edges_dict.items():
        edge_attr.append(v)
        print(f'{k},{len(v)}')
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    edge_index_list = torch.tensor([edge for edge in edges_dict.keys()], dtype=torch.int64).t().contiguous()
    edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)

    d = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    num_classes = len(d)
    x = torch.nn.functional.one_hot(
        torch.tensor([d[n] for n in row['seq']], dtype=torch.long),
        num_classes=num_classes
    ).to(torch.float32)

    y = torch.tensor([row['is_positive']], dtype=torch.int64)
    graph = Data(edge_index=edge_index_list, edge_attr=edge_attr, y=y, x=x)
    return graph


def get_all_indexes_from_string(string):
    return [int(char) for char in string if char.isdigit()]


def get_graph_hot_encoding_continuity(row, cols):
    edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [],
                  (1,2): [], (1,3): [], (1,4): [], (1,5): [],
                  (2,3): [], (2,4): [], (2,5): [],
                  (3,4): [], (3,5): [],
                  (4,5): []}

    for i in range(len(cols)):
        weigth = row[i]
        cols_indexes = get_all_indexes_from_string(cols[i])
        if len(cols_indexes) == 2 and 7 not in cols_indexes:
            edges_dict[(cols_indexes[0]-1, cols_indexes[1]-1)].append(weigth)

    edge_attr = []
    edge_index_list = []

    for (i, j), weights in edges_dict.items():
        if len(weights) == 0:
            continue
        is_consecutive = 1.0 if abs(i - j) == 1 else 0.0
        avg_weight = sum(weights) / len(weights)
        edge_attr.append(weights + [is_consecutive])
        edge_index_list.append([i, j])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_index_list = torch.tensor(edge_index_list, dtype=torch.int64).t().contiguous()

    edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)
    edge_attr_symmetric = torch.cat([edge_attr, edge_attr], dim=0)

    d = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    num_classes = len(d)
    x = torch.nn.functional.one_hot(
        torch.tensor([d[n] for n in row['seq']], dtype=torch.long),
        num_classes=num_classes
    ).to(torch.float32)

    y = torch.tensor([row['is_positive']], dtype=torch.int64)
    graph = Data(edge_index=edge_index_symmetric, edge_attr=edge_attr_symmetric, y=y, x=x)
    return graph


def get_graph_hot_encoding_v3(row, cols):
    NUM_NODES = 6
    d = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    num_classes = len(d)

    seq = row['seq'][:NUM_NODES]
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


def display_graph_and_weights(graph):
    print("Edge Index:")
    print(graph.edge_index)
    print("\nEdge Attributes (weights):")
    print(graph.edge_attr)


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


def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()


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
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def plot_model_metrics_during_training(epoch_data, model_name, fold_number=None, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'{model_name} Metrics During Training{f" - Fold {fold_number}" if fold_number else ""}',
        fontsize=14, fontweight='bold'
    )

    epochs = epoch_data.get('epochs', [])

    ax = axes[0]
    if 'train_acc' in epoch_data:
        ax.plot(epochs, epoch_data['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    if 'test_acc' in epoch_data:
        ax.plot(epochs, epoch_data['test_acc'], 'r-', label='Test Accuracy', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'train_f1' in epoch_data:
        ax.plot(epochs, epoch_data['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o', markersize=3)
    if 'test_f1' in epoch_data:
        ax.plot(epochs, epoch_data['test_f1'], 'r-', label='Test F1', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if 'train_mcc' in epoch_data:
        ax.plot(epochs, epoch_data['train_mcc'], 'b-', label='Train MCC', linewidth=2, marker='o', markersize=3)
    if 'test_mcc' in epoch_data:
        ax.plot(epochs, epoch_data['test_mcc'], 'r-', label='Test MCC', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MCC')
    ax.set_title('Matthews Correlation Coefficient over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")

    return fig


def evaluate_classifiers(X_train, y_train, X_val, y_val, prefix=''):
    results = {}

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_val)
    results['GaussianNB'] = {
        'accuracy':  accuracy_score(y_val, y_pred_gnb),
        'precision': precision_score(y_val, y_pred_gnb),
        'recall':    recall_score(y_val, y_pred_gnb),
        'f1':        f1_score(y_val, y_pred_gnb)
    }
    gnb_metrics = {
        'epochs':    [1],
        'train_acc': [accuracy_score(y_train, gnb.predict(X_train))],
        'test_acc':  [results['GaussianNB']['accuracy']],
        'train_f1':  [f1_score(y_train, gnb.predict(X_train), zero_division=0)],
        'test_f1':   [results['GaussianNB']['f1']],
        'train_mcc': [matthews_corrcoef(y_train, gnb.predict(X_train))],
        'test_mcc':  [matthews_corrcoef(y_val, y_pred_gnb)]
    }
    plot_model_metrics_during_training(gnb_metrics, 'GaussianNB', save_path=f'gnb_metrics_{prefix}.png')

    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled   = scaler_fold.transform(X_val)
    clf = svm.SVC()
    clf.fit(X_train_scaled, y_train)
    y_pred_svm = clf.predict(X_val_scaled)
    results['SVM'] = {
        'accuracy':  accuracy_score(y_val, y_pred_svm),
        'precision': precision_score(y_val, y_pred_svm),
        'recall':    recall_score(y_val, y_pred_svm),
        'f1':        f1_score(y_val, y_pred_svm)
    }
    svm_metrics = {
        'epochs':    [1],
        'train_acc': [accuracy_score(y_train, clf.predict(X_train_scaled))],
        'test_acc':  [results['SVM']['accuracy']],
        'train_f1':  [f1_score(y_train, clf.predict(X_train_scaled), zero_division=0)],
        'test_f1':   [results['SVM']['f1']],
        'train_mcc': [matthews_corrcoef(y_train, clf.predict(X_train_scaled))],
        'test_mcc':  [matthews_corrcoef(y_val, y_pred_svm)]
    }
    plot_model_metrics_during_training(svm_metrics, 'SVM', save_path=f'svm_metrics_{prefix}.png')

    print(f"\nEvaluation results ({prefix}):")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")
    return results


def main():
    set_all_seeds(SEED)

    # ── Load raw data ────────────────────────────────────────────────────────
    dpos = pd.read_csv("positve.csv", sep=',', index_col=None)
    dneg = pd.read_csv("negative.csv", sep=',', index_col=None)
    data_full = pd.concat([dneg, dpos])

    seqs = []
    max_len = data_full.shape[0]
    read_sequences(seqs, "positve_seq.csv", max_len)
    read_sequences(seqs, "negative_seq.csv", max_len)
    data_full['seq'] = seqs

    # ── Final eval dataframe (mmcif) ─────────────────────────────────────────
    dftofinalevaltmp = pd.read_csv("filtered_geometric_features_to_test8.csv", sep=',', index_col=0)
    dftofinaleval_X  = dftofinalevaltmp.drop(columns=['is_positive'])
    dftofinaleval_Y  = dftofinalevaltmp['is_positive']
    graph_dftofinaleval = dftofinalevaltmp.apply(
        lambda x: get_graph_hot_encoding_v3(x, dftofinalevaltmp.columns[:-1]), axis=1
    )
    final_eval_loader = DataLoader(graph_dftofinaleval, batch_size=32)

    # ── Load & filter main dataset ───────────────────────────────────────────
    dftofilter = pd.read_csv("filtered_geometric_features_bez1JID.csv", sep=',', index_col=0)
    num_gnra_pre_filter = dftofilter["gnra"].value_counts()

    clusters_path = "clusters.json"
    all_cluster_members = []
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r') as f:
            clusters_data = json.load(f)
        if isinstance(clusters_data, dict):
            if 'clusters' in clusters_data and isinstance(clusters_data['clusters'], list):
                clusters_list = clusters_data['clusters']
            else:
                clusters_list = [v for v in clusters_data.values() if isinstance(v, dict)]
        elif isinstance(clusters_data, list):
            clusters_list = clusters_data
        else:
            clusters_list = []

        for cluster in clusters_list:
            members = cluster.get('members') if isinstance(cluster, dict) else None
            if isinstance(members, list):
                all_cluster_members.extend([str(m) for m in members])
            elif isinstance(members, str):
                parts = [s.strip() for s in members.split(',') if s.strip()]
                all_cluster_members.extend(parts)
    else:
        print(f"clusters.json not found at {clusters_path}")

    print(f"Total cluster members collected: {len(all_cluster_members)}")
    print(f"Sample cluster members: {all_cluster_members[:10]}")

    all_cluster_members = [m.replace('.cif', '') for m in all_cluster_members]
    print(dftofilter)
    dfpreresample = dftofilter[~dftofilter.index.isin(all_cluster_members)]
    print(dfpreresample)
    num_gnra_post_filter = dfpreresample["gnra"].value_counts()

    # ── Resample ─────────────────────────────────────────────────────────────
    iks   = dfpreresample.drop(columns=['gnra'])
    igrek = dfpreresample['gnra']
    cnn_sampler = CondensedNearestNeighbour(random_state=42)
    cnn_sampler.fit_resample(iks, igrek)
    df = dfpreresample.iloc[cnn_sampler.sample_indices_]

    stat, p_value = shapiro(df)
    print(f'Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}')

    # ── Split by date ─────────────────────────────────────────────────────────
    pre, post = filter_script.filter_pandas_dataframe_by_date(
        df, 'rna_pdb_release_dates.csv', '2024-10-20T00:00:00+0000'
    )
    pre_indices  = pre.index.copy()
    post_indices = post.index.copy()

    print(f"number of rows in df: {df.shape[0]}")
    print("=" * 60 + " DF PRE selected date " + "=" * 60)
    print(pre)
    print("=" * 60 + " DF POST selected date " + "=" * 60)
    print(post)
    print("=" * 60 + " END " + "=" * 60)

    num_gnra_in_post_df     = post["gnra"].value_counts()
    num_all_files_in_post_df = post.shape[0]
    num_gnra_in_pre_df      = pre["gnra"].value_counts()
    num_all_files_in_pre_df  = pre.shape[0]

    y     = df['gnra']
    y_pre  = pre['gnra']
    y_post = post['gnra']

    df   = df.drop(columns=['gnra'])
    pre  = pre.drop(columns=['gnra'])
    post = post.drop(columns=['gnra'])

    # ── Classical classifiers (GNB + SVM) ────────────────────────────────────
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nUsing {n_splits}-fold stratified cross-validation on pre, validating on post.")

    pre   = pre.reset_index(drop=True)
    post  = post.reset_index(drop=True)
    y_pre  = y_pre.reset_index(drop=True)
    y_post = y_post.reset_index(drop=True)

    cv_results = evaluate_classifiers(pre, y_pre, post, y_post, prefix='pre_post')

    # ── Build graph DataFrames ────────────────────────────────────────────────
    scaler     = StandardScaler()
    df_std     = pd.DataFrame(scaler.fit_transform(df))
    df_std.columns = df.columns

    seqs_series = pd.Series(seqs, name='seq')
    if len(seqs_series) > df_std.shape[0]:
        seqs_series = seqs_series.iloc[:df_std.shape[0]].reset_index(drop=True)
    elif len(seqs_series) < df_std.shape[0]:
        seqs_series = pd.Series([''] * df_std.shape[0], name='seq')

    df_graph = df_std.reset_index(drop=True).copy()
    df_graph['seq']         = seqs_series
    df_graph['is_positive'] = y.reset_index(drop=True).astype(int)

    df_d = df.copy()
    df_d['is_positive'] = y

    # ── GNN — k-fold on pre, validate on post ────────────────────────────────
    print("\nUsing StratifiedKFold on pre for GNN training/evaluation, with post as validation set...")

    scaler_pre = StandardScaler()
    pre_std    = pd.DataFrame(scaler_pre.fit_transform(pre), columns=pre.columns)

    scaler_post = StandardScaler()
    post_std    = pd.DataFrame(scaler_post.fit_transform(post), columns=post.columns)

    df_graph_pre = pre_std.copy()
    df_graph_pre['seq']         = [seqs[i] for i in pre_indices]  if hasattr(pre_indices,  '__iter__') else [''] * len(pre_std)
    df_graph_pre['is_positive'] = y_pre.values

    df_graph_post = post_std.copy()
    df_graph_post['seq']         = [seqs[i] for i in post_indices] if hasattr(post_indices, '__iter__') else [''] * len(post_std)
    df_graph_post['is_positive'] = y_post.values

    print("df_graph_pre class balance:",  df_graph_pre['is_positive'].value_counts())
    print("df_graph_post class balance:", df_graph_post['is_positive'].value_counts())
    print(f"previously gathered data \n gnra in post: {num_gnra_in_post_df} len post: {num_all_files_in_post_df} \n gnra in pre: {num_gnra_in_pre_df} len pre: {num_all_files_in_pre_df}")
    print(f"Total graphs in pre: {len(df_graph_pre)}")
    print(f"Total graphs in post (fixed validation): {len(df_graph_post)}")

    cols_post = df_graph_post.columns[:-1]
    post_test_dataset = df_graph_post.apply(lambda x: get_graph_hot_encoding_v3(x, cols_post), axis=1)
    post_test_loader  = DataLoader(post_test_dataset, batch_size=32)

    print("\nUsing StratifiedKFold for GNN training/evaluation...")
    print("Graph DataFrame head:=======================================================")
    print(df_graph)
    print("Graph DataFrame columns:")
    print(df_graph.columns)

    gnn_fold_results = []
    USE_FULL_DATASET = False

    if USE_FULL_DATASET:
        fold_splits = [(df_graph_pre.index.tolist(), df_graph_pre.index.tolist())]
    else:
        fold_splits = list(skf.split(df_graph_pre.drop(columns=['is_positive']), df_graph_pre['is_positive']))

    fold_index = 0
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n--- GNN Fold {fold + 1}/{n_splits} ---")
        set_all_seeds(SEED)

        df_train = df_graph_pre.iloc[train_idx].reset_index(drop=True)
        df_test  = df_graph_pre.iloc[val_idx].reset_index(drop=True)

        pos = (df_train['is_positive'] == 1).sum()
        neg = (df_train['is_positive'] == 0).sum()
        print(f"NEG: {neg} POS: {pos}")
        pos_weight_value = neg / pos
        print(f"Positives: {pos}, Negatives: {neg}, Pos weight: {pos_weight_value:.2f}x")

        class_weights = torch.tensor([1.0, pos_weight_value], dtype=torch.float)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        cols = df_train.columns[:-1]
        train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)
        test_dataset  = df_test.apply(lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=32)

        model     = GCN(hidden_channels=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        best_model_state        = None
        best_model_state_by_acc = None
        best_model_state_by_mcc = None
        best_acc = 0.0
        best_f1  = 0.0
        best_mcc = 0.0
        no_improve_counter = 0
        max_no_improve     = 30

        epoch_metrics = {
            'epochs': [], 'train_acc': [], 'test_acc': [],
            'train_f1': [], 'test_f1': [], 'train_mcc': [], 'test_mcc': []
        }

        max_epochs = 200
        for epoch in range(1, max_epochs + 1):
            train(model, train_loader, optimizer, criterion)

            train_acc, train_preds, train_labels = test(model, train_loader, return_predictions=True)
            test_acc,  test_preds,  test_labels  = test(model, test_loader,  return_predictions=True)

            if epoch == 40:
                print(f"=======================================================================\n {train_labels} \n{train_preds}")

            train_f1  = f1_score(train_labels, train_preds, zero_division=0)
            test_f1   = f1_score(test_labels,  test_preds,  zero_division=0)
            train_mcc = matthews_corrcoef(train_labels, train_preds)
            test_mcc  = matthews_corrcoef(test_labels,  test_preds)

            epoch_metrics['epochs'].append(epoch)
            epoch_metrics['train_acc'].append(train_acc)
            epoch_metrics['test_acc'].append(test_acc)
            epoch_metrics['train_f1'].append(train_f1)
            epoch_metrics['test_f1'].append(test_f1)
            epoch_metrics['train_mcc'].append(train_mcc)
            epoch_metrics['test_mcc'].append(test_mcc)

            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model_state = copy.deepcopy(model.state_dict())
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_state_by_acc = copy.deepcopy(model.state_dict())
            if test_mcc > best_mcc:
                best_mcc = test_mcc
                best_model_state_by_mcc = copy.deepcopy(model.state_dict())
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f}')

            if no_improve_counter >= max_no_improve:
                print("Early stopping due to no improvement")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        plot_model_metrics_during_training(epoch_metrics, 'GNN', fold_number=fold+1,
                                           save_path=f'gnn_training_metrics_fold_{fold+1}.png')
        model.eval()

        # Evaluate on post holdout
        print("\n--- Final evaluation on post (holdout) dataset ---")
        print("\n -- model by f1 --")
        model.load_state_dict(best_model_state)
        final_acc, final_preds, final_labels = test(model, post_test_loader, return_predictions=True)
        print(f"Post holdout Acc: {final_acc:.4f}")
        print(f"Post holdout F1:  {f1_score(final_labels, final_preds, zero_division=0):.4f}")
        print(f"Post holdout MCC: {matthews_corrcoef(final_labels, final_preds):.4f}")

        print("\n -- model by accuracy --")
        model.load_state_dict(best_model_state_by_acc)
        final_acc, final_preds, final_labels = test(model, post_test_loader, return_predictions=True)
        print(f"Post holdout Acc: {final_acc:.4f}")
        print(f"Post holdout F1:  {f1_score(final_labels, final_preds, zero_division=0):.4f}")
        print(f"Post holdout MCC: {matthews_corrcoef(final_labels, final_preds):.4f}")

        print("\n -- model by mcc --")
        model.load_state_dict(best_model_state_by_mcc)
        final_acc, final_preds, final_labels = test(model, post_test_loader, return_predictions=True)
        print(f"Post holdout Acc: {final_acc:.4f}")
        print(f"Post holdout F1:  {f1_score(final_labels, final_preds, zero_division=0):.4f}")
        print(f"Post holdout MCC: {matthews_corrcoef(final_labels, final_preds):.4f}")

        final_acc2, final_preds2, final_labels2 = test(model, final_eval_loader, return_predictions=True)
        final_probs, final_labels_probs = test_proba(model, final_eval_loader)
        cmascore = cma.customAssesment2(final_labels2, final_preds2)
        print(f"mmcif Acc: {final_acc2:.4f}")
        print(f"mmcif F1:  {f1_score(final_labels2, final_preds2, zero_division=0):.4f}")
        print(f"mmcif MCC: {matthews_corrcoef(final_labels2, final_preds2):.4f}")
        print(f"real labels:  {dftofinaleval_Y.values}")
        print(f"GNN labels:   {final_labels2}")
        print(f"GNN predictions: {final_preds2}")
        print(f"Post holdout probabilities (class 1): {np.round(final_probs, 3)}")
        print(f"Custom model assessment {cmascore}")

        torch.save(best_model_state_by_mcc, f'model_best_mcc{fold_index}.pth')
        if cmascore > SCORE_THRESHOLD:
            print(f"Custom model assessment score {cmascore} exceeds threshold {SCORE_THRESHOLD}, saving model.")
            torch.save(best_model_state_by_mcc, f'{FOLDER}/model_best_mcc_cma{SEED}_{fold_index}.pth')
        fold_index += 1

    # ── Post-loop diagnostics (uses last fold's test_loader/model) ────────────
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            out  = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    try:
        print('y_true dist:', np.bincount(y_true) if len(y_true) > 0 else 'empty')
    except Exception:
        print('y_true dist: could not compute bincount')
    try:
        print('y_pred dist:', np.bincount(y_pred) if len(y_pred) > 0 else 'empty')
    except Exception:
        print('y_pred dist: could not compute bincount')
    try:
        print('unique preds:', np.unique(y_pred, return_counts=True))
    except Exception:
        pass
    try:
        print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
    except Exception:
        print('confusion matrix: failed')
    try:
        print('classification report:\n', classification_report(y_true, y_pred, zero_division=0))
    except Exception as e:
        print('classification report: failed', e)

    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1  = f1_score(y_true, y_pred, zero_division=0)
    fold_mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Fold {fold + 1} best Test Acc: {best_acc:.4f}")
    print(f"Fold {fold + 1} metrics: Acc={fold_acc:.4f}, F1={fold_f1:.4f}, MCC={fold_mcc:.4f}")
    gnn_fold_results.append({'accuracy': fold_acc, 'f1': fold_f1, 'mcc': fold_mcc})

    # ── Final GNN CV summary ──────────────────────────────────────────────────
    print("\nGNN cross-validation results:")
    accs = [d['accuracy'] for d in gnn_fold_results]
    f1s  = [d['f1']       for d in gnn_fold_results]
    mccs = [d['mcc']      for d in gnn_fold_results]
    print(f"  Per-fold accuracy: {accs}")
    print(f"  Per-fold F1:       {f1s}")
    print(f"  Per-fold MCC:      {mccs}")
    print(f"  Mean accuracy: {np.mean(accs):.4f} (std: {np.std(accs):.4f})")
    print(f"  Mean F1:       {np.mean(f1s):.4f} (std: {np.std(f1s):.4f})")
    print(f"  Mean MCC:      {np.mean(mccs):.4f} (std: {np.std(mccs):.4f})")


if __name__ == "__main__":
    main()
    #pass