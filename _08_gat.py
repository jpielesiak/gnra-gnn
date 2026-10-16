import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
import csv
from itertools import combinations
import json, os
from imblearn.under_sampling import CondensedNearestNeighbour

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
from torch.nn import Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import copy
from torch_geometric.nn import GATv2Conv
import random
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score

NODE_DIM = 49  # 4 one-hot + angle + torsion features
EDGE_DIM = 2   # distance value + is_consecutive flag
SEED = 42
FOLDER = "test_models_annealing"
SCORE_THRESHOLD = -9

# ── Optuna settings ────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS   = 30   # number of hyperparameter combinations to try
OPTUNA_N_FOLDS    = 3    # inner CV folds used during each Optuna trial (faster than 5)
OPTUNA_MAX_EPOCHS = 80   # shorter budget per trial to keep search tractable


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


def get_all_indexes_from_string(string):
    return [int(char) for char in string if char.isdigit()]


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


# ── Parameterised GCN ─────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    """
    GATv2-based GNN whose architecture is fully controlled by the `hp` dict
    produced by Optuna (or a set of manual defaults).

    hp keys
    -------
    hidden1        : output channels of conv1
    hidden2        : output channels of conv2
    hidden3        : output channels of conv3
    heads          : number of attention heads (all layers share the same value)
    dropout        : dropout probability before the final linear layer
    """

    def __init__(self, hp: dict):
        super().__init__()
        h1      = hp['hidden1']
        h2      = hp['hidden2']
        h3      = hp['hidden3']
        heads   = hp['heads']
        # GATv2Conv output dim = out_channels * heads (when concat=True, the default)
        self.conv1 = GATv2Conv(NODE_DIM,    h1, edge_dim=EDGE_DIM, heads=heads,  concat=True)
        self.conv2 = GATv2Conv(h1 * heads,  h2, edge_dim=EDGE_DIM, heads=heads,  concat=True)
        self.conv3 = GATv2Conv(h2 * heads,  h3, edge_dim=EDGE_DIM, heads=1,      concat=False)
        self.lin   = torch.nn.Linear(h3, 2)
        self.dropout = hp['dropout']

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# ── Default hyperparameters (used if you skip Optuna) ─────────────────────────

DEFAULT_HP = {
    'hidden1':     32,
    'hidden2':     64,
    'hidden3':     32,
    'heads':        1,
    'dropout':     0.3,
    'lr':          1e-3,
    'weight_decay': 0.0,
    'batch_size':  32,
}


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
        ax.plot(epochs, epoch_data['test_acc'],  'r-', label='Test Accuracy',  linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy over Epochs')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'train_f1' in epoch_data:
        ax.plot(epochs, epoch_data['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o', markersize=3)
    if 'test_f1' in epoch_data:
        ax.plot(epochs, epoch_data['test_f1'],  'r-', label='Test F1',  linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score'); ax.set_title('F1 Score over Epochs')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if 'train_mcc' in epoch_data:
        ax.plot(epochs, epoch_data['train_mcc'], 'b-', label='Train MCC', linewidth=2, marker='o', markersize=3)
    if 'test_mcc' in epoch_data:
        ax.plot(epochs, epoch_data['test_mcc'],  'r-', label='Test MCC',  linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MCC'); ax.set_title('Matthews Correlation Coefficient over Epochs')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    return fig

def correct_prior(probs_positive, train_prevalence, real_prevalence):
    """
    Shifts model output probabilities from training prior to real-world prior.
    probs_positive : np.array of P(y=1|x) from the model
    train_prevalence: fraction of positives in training data (e.g. 0.30)
    real_prevalence : true real-world fraction (e.g. 0.01)
    """
    # Recover likelihood ratio from training posterior
    odds_train = probs_positive / (1 - probs_positive + 1e-9)
    likelihood_ratio = odds_train * ((1 - train_prevalence) / train_prevalence)

    # Apply real-world prior
    odds_real = likelihood_ratio * (real_prevalence / (1 - real_prevalence))
    return odds_real / (1 + odds_real)


def find_best_threshold(probs, labels, steps=200):
    best_thr, best_mcc = 0.5, -2.0
    for thr in np.linspace(0.01, 0.99, steps):
        preds = (probs >= thr).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    return best_thr, best_mcc


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

    scaler_fold    = StandardScaler()
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


# ── Optuna objective ───────────────────────────────────────────────────────────

def run_single_gnn_fold(df_train, df_val, hp, max_epochs, pos_weight_value, seed):
    """
    Train one GNN fold with the given hyperparams and return the best val MCC.
    Kept separate so both the Optuna objective and the main loop can call it.
    """
    set_all_seeds(seed)

    class_weights = torch.tensor([1.0, pos_weight_value], dtype=torch.float)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

    cols          = df_train.columns[:-1]
    train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)
    val_dataset   = df_val.apply(  lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)

    train_loader  = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=hp['batch_size'])

    model     = GCN(hp)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )

    best_mcc          = -2.0
    no_improve        = 0
    max_no_improve    = 20  # tighter patience during search to keep trials fast

    for epoch in range(1, max_epochs + 1):
        train(model, train_loader, optimizer, criterion)
        _, val_preds, val_labels = test(model, val_loader, return_predictions=True)
        val_mcc = matthews_corrcoef(val_labels, val_preds)

        if val_mcc > best_mcc:
            best_mcc   = val_mcc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            break
    # probs = test_proba(model, val_loader)[0]
    # return average_precision_score(val_labels, probs) 
    return best_mcc


def build_optuna_objective(df_graph_pre, seed):
    """
    Returns a closure that Optuna can call as objective(trial).
    Uses OPTUNA_N_FOLDS inner folds on df_graph_pre; optimises mean MCC.
    """
    inner_skf = StratifiedKFold(
        n_splits=OPTUNA_N_FOLDS, shuffle=True, random_state=seed
    )

    def objective(trial):
        # ── Search space ──────────────────────────────────────────────────────
        hp = {
            # Architecture
            'hidden1':      trial.suggest_categorical('hidden1',      [16, 32, 64]),
            'hidden2':      trial.suggest_categorical('hidden2',      [32, 64, 128]),
            'hidden3':      trial.suggest_categorical('hidden3',      [16, 32, 64]),
            'heads':        trial.suggest_categorical('heads',        [1, 2, 4]),
            'dropout':      trial.suggest_float(      'dropout',       0.1, 0.5, step=0.1),
            # Optimiser
            'lr':           trial.suggest_float(      'lr',            1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float(      'weight_decay',  0.0,  1e-3, step=1e-4),
            'batch_size':   trial.suggest_categorical('batch_size',   [16, 32, 64]),
        }

        fold_mccs = []
        for train_idx, val_idx in inner_skf.split(
            df_graph_pre.drop(columns=['is_positive']),
            df_graph_pre['is_positive']
        ):
            df_train = df_graph_pre.iloc[train_idx].reset_index(drop=True)
            df_val   = df_graph_pre.iloc[val_idx].reset_index(drop=True)

            pos              = (df_train['is_positive'] == 1).sum()
            neg              = (df_train['is_positive'] == 0).sum()
            pos_weight_value = neg / pos if pos > 0 else 1.0

            mcc = run_single_gnn_fold(
                df_train, df_val, hp,
                max_epochs=OPTUNA_MAX_EPOCHS,
                pos_weight_value=pos_weight_value,
                seed=seed,
            )
            fold_mccs.append(mcc)

            # Optuna pruning: report intermediate value after each inner fold
            trial.report(np.mean(fold_mccs), step=len(fold_mccs))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_mccs))

    return objective


def run_optuna_search(df_graph_pre, seed, n_trials=OPTUNA_N_TRIALS):
    """
    Run Optuna hyperparameter search and return the best hp dict.
    """
    print(f"\n{'='*60}")
    print(f"  Starting Optuna search: {n_trials} trials, "
          f"{OPTUNA_N_FOLDS} inner folds each, "
          f"up to {OPTUNA_MAX_EPOCHS} epochs/fold")
    print(f"{'='*60}\n")

    sampler = TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='gnn_rna_motif',
    )

    objective = build_optuna_objective(df_graph_pre, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna search complete.")
    print(f"  Best trial:  #{study.best_trial.number}")
    print(f"  Best MCC:    {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:20s}: {v}")

    # Visualise importance if plotly is available
    try:
        fig_imp = optuna.visualization.plot_param_importances(study)
        fig_imp.write_image("optuna_param_importances.png")
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.write_image("optuna_optimization_history.png")
        print("  Saved optuna_param_importances.png and optuna_optimization_history.png")
    except Exception:
        pass

    return study.best_params, study


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)

    # ── Load raw data ─────────────────────────────────────────────────────────
    dpos = pd.read_csv("positve.csv",    sep=',', index_col=None)
    dneg = pd.read_csv("negative.csv",   sep=',', index_col=None)
    data_full = pd.concat([dneg, dpos])

    seqs    = []
    max_len = data_full.shape[0]
    read_sequences(seqs, "positve_seq.csv",  max_len)
    read_sequences(seqs, "negative_seq.csv", max_len)
    data_full['seq'] = seqs

    # ── Final eval dataframe (mmcif) ──────────────────────────────────────────
    dftofinalevaltmp = pd.read_csv("filtered_geometric_features_to_test8a.csv", sep=',', index_col=0)
    dftofinaleval_X  = dftofinalevaltmp.drop(columns=['is_positive'])
    dftofinaleval_Y  = dftofinalevaltmp['is_positive']
    graph_dftofinaleval = dftofinalevaltmp.apply(
        lambda x: get_graph_hot_encoding_v3(x, dftofinalevaltmp.columns[:-1]), axis=1
    )
    final_eval_loader = DataLoader(graph_dftofinaleval, batch_size=32)

    # ── Load & filter main dataset ────────────────────────────────────────────
    dftofilter = pd.read_csv("filtered_geometric_features_bez1JID.csv", sep=',', index_col=0)

    clusters_path       = "clusters.json"
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

    all_cluster_members = [m.replace('.cif', '') for m in all_cluster_members]
    dfpreresample = dftofilter[~dftofilter.index.isin(all_cluster_members)]

    # ── Resample ──────────────────────────────────────────────────────────────
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

    y_pre  = pre['gnra']
    y_post = post['gnra']
    df     = df.drop(columns=['gnra'])
    pre    = pre.drop(columns=['gnra'])
    post   = post.drop(columns=['gnra'])

    # ── Classical classifiers ─────────────────────────────────────────────────
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pre   = pre.reset_index(drop=True)
    post  = post.reset_index(drop=True)
    y_pre  = y_pre.reset_index(drop=True)
    y_post = y_post.reset_index(drop=True)

    cv_results = evaluate_classifiers(pre, y_pre, post, y_post, prefix='pre_post')

    # ── Build graph DataFrames ────────────────────────────────────────────────
    scaler_pre  = StandardScaler()
    pre_std     = pd.DataFrame(scaler_pre.fit_transform(pre),   columns=pre.columns)
    scaler_post = StandardScaler()
    post_std    = pd.DataFrame(scaler_post.fit_transform(post), columns=post.columns)

    df_graph_pre              = pre_std.copy()
    df_graph_pre['seq']       = [seqs[i] for i in pre_indices]  if hasattr(pre_indices,  '__iter__') else [''] * len(pre_std)
    df_graph_pre['is_positive'] = y_pre.values

    df_graph_post               = post_std.copy()
    df_graph_post['seq']        = [seqs[i] for i in post_indices] if hasattr(post_indices, '__iter__') else [''] * len(post_std)
    df_graph_post['is_positive'] = y_post.values

    print("df_graph_pre class balance:",  df_graph_pre['is_positive'].value_counts())
    print("df_graph_post class balance:", df_graph_post['is_positive'].value_counts())

    cols_post         = df_graph_post.columns[:-1]
    post_test_dataset = df_graph_post.apply(lambda x: get_graph_hot_encoding_v3(x, cols_post), axis=1)
    post_test_loader  = DataLoader(post_test_dataset, batch_size=32)

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    # The search runs on the pre-split data only — post remains a true holdout.
    best_params, optuna_study = run_optuna_search(
        df_graph_pre, seed=SEED, n_trials=OPTUNA_N_TRIALS
    )

    # Merge best Optuna params with any keys that Optuna doesn't tune (none here,
    # but kept as a safety net in case DEFAULT_HP grows).
    best_hp = {**DEFAULT_HP, **best_params}
    print(f"\nFinal hyperparameters for main training: {best_hp}")

    # ── GNN — k-fold on pre, validate on post (using best_hp) ─────────────────
    print("\nRunning full k-fold GNN training with best hyperparameters...")
    fold_splits    = list(skf.split(
        df_graph_pre.drop(columns=['is_positive']), df_graph_pre['is_positive']
    ))
    gnn_fold_results = []
    fold_index       = 0

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n--- GNN Fold {fold + 1}/{n_splits} ---")
        set_all_seeds(SEED)

        df_train = df_graph_pre.iloc[train_idx].reset_index(drop=True)
        df_val   = df_graph_pre.iloc[val_idx].reset_index(drop=True)

        pos              = (df_train['is_positive'] == 1).sum()
        neg              = (df_train['is_positive'] == 0).sum()
        pos_weight_value = neg / pos if pos > 0 else 1.0
        print(f"NEG: {neg}  POS: {pos}  pos_weight: {pos_weight_value:.2f}x")

        #CHANGED THIS TO THE MORE REALISTIC CLASS WEIGHT BASED ON THE EXPECTED REAL-WORLD POSITIVE RATE
        #class_weights = torch.tensor([1.0, pos_weight_value], dtype=torch.float)
        REAL_WORLD_POSITIVE_RATE = 0.01  # 1%
        real_world_ratio = (1 - REAL_WORLD_POSITIVE_RATE) / REAL_WORLD_POSITIVE_RATE  # ≈ 99

        class_weights = torch.tensor([1.0, real_world_ratio], dtype=torch.float)
        
        criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

        cols          = df_train.columns[:-1]
        train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)
        val_dataset   = df_val.apply(  lambda x: get_graph_hot_encoding_v3(x, cols), axis=1)

        train_loader = DataLoader(train_dataset, batch_size=best_hp['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=best_hp['batch_size'])

        model     = GCN(best_hp)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_hp['lr'], weight_decay=best_hp['weight_decay']
        )

        best_model_state        = None
        best_model_state_by_acc = None
        best_model_state_by_mcc = None
        best_acc       = 0.0
        best_f1        = 0.0
        best_mcc       = 0.0
        no_improve     = 0
        max_no_improve = 30

        epoch_metrics = {
            'epochs': [], 'train_acc': [], 'test_acc': [],
            'train_f1': [], 'test_f1': [], 'train_mcc': [], 'test_mcc': []
        }

        max_epochs = 200
        for epoch in range(1, max_epochs + 1):
            train(model, train_loader, optimizer, criterion)

            train_acc, train_preds, train_labels = test(model, train_loader, return_predictions=True)
            val_acc,   val_preds,   val_labels   = test(model, val_loader,   return_predictions=True)

            train_f1  = f1_score(train_labels, train_preds, zero_division=0)
            val_f1    = f1_score(val_labels,   val_preds,   zero_division=0)
            train_mcc = matthews_corrcoef(train_labels, train_preds)
            val_mcc   = matthews_corrcoef(val_labels,   val_preds)

            epoch_metrics['epochs'].append(epoch)
            epoch_metrics['train_acc'].append(train_acc)
            epoch_metrics['test_acc'].append(val_acc)
            epoch_metrics['train_f1'].append(train_f1)
            epoch_metrics['test_f1'].append(val_f1)
            epoch_metrics['train_mcc'].append(train_mcc)
            epoch_metrics['test_mcc'].append(val_mcc)

            if val_f1 > best_f1:
                best_f1          = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
            if val_acc > best_acc:
                best_acc                = val_acc
                best_model_state_by_acc = copy.deepcopy(model.state_dict())
            if val_mcc > best_mcc:
                best_mcc                = val_mcc
                best_model_state_by_mcc = copy.deepcopy(model.state_dict())
                no_improve              = 0
            else:
                no_improve += 1

            print(f'Epoch {epoch:03d} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} '
                  f'| Val F1 {val_f1:.4f} | Val MCC {val_mcc:.4f}')

            if no_improve >= max_no_improve:
                print("Early stopping.")
                break

        plot_model_metrics_during_training(
            epoch_metrics, 'GNN', fold_number=fold + 1,
            save_path=f'gnn_training_metrics_fold_{fold+1}.png'
        )

        # ── Evaluate on post holdout ──────────────────────────────────────────
        print("\n--- Final evaluation on post (holdout) ---")
        for label, state in [
            ('best F1',  best_model_state),
            ('best Acc', best_model_state_by_acc),
            ('best MCC', best_model_state_by_mcc),
        ]:
            if state is None:
                continue
            model.load_state_dict(state)
            final_acc, final_preds, final_labels = test(model, post_test_loader, return_predictions=True)
            print(f"  [{label}] Acc {final_acc:.4f} | "
                  f"F1 {f1_score(final_labels, final_preds, zero_division=0):.4f} | "
                  f"MCC {matthews_corrcoef(final_labels, final_preds):.4f}")

        # mmcif evaluation (best-MCC model)
        model.load_state_dict(best_model_state_by_mcc)
        final_acc2, final_preds2, final_labels2 = test(model, final_eval_loader, return_predictions=True)
        final_probs, final_labels_probs                            = test_proba(model, final_eval_loader)
        # In your evaluation block, after test_proba():
        final_probs_corrected = correct_prior(
            final_probs,
            train_prevalence=0.30,  # what your training set looks like
            real_prevalence=0.01    # what reality looks like
        )
        best_thr, best_mcc = find_best_threshold(final_probs_corrected, final_labels_probs)
        print(f"Best threshold: {best_thr:.3f}  MCC: {best_mcc:.4f}")
        final_preds_corrected = (final_probs_corrected >= best_thr).astype(int)
        cmascore = cma.customAssesment2(final_labels2, final_preds2)
        cmascorecorrected = cma.customAssesment2(final_labels2, final_preds_corrected)
        print(f"\nmmcif Acc {final_acc2:.4f} | "
              f"F1 {f1_score(final_labels2, final_preds2, zero_division=0):.4f} | "
              f"MCC {matthews_corrcoef(final_labels2, final_preds2):.4f} | "
              f"CMA {cmascore}")
        print(f"GNN labels:   {final_labels2}")
        print(f"GNN predictions: {final_preds2}")
        print(f"GNN preds corrected: {final_preds_corrected}")
        print(f"GNN score corrected: {cmascorecorrected}")
        torch.save(best_model_state_by_mcc, f'model_best_mcc{fold_index}.pth')
        if cmascore > SCORE_THRESHOLD:
            print(f"CMA score {cmascore} > threshold {SCORE_THRESHOLD} — saving model.")
            torch.save(best_model_state_by_mcc,
                       f'{FOLDER}/model_best_mcc_cma{SEED}_{fold_index}.pth')
        fold_index += 1

        # Collect fold metrics (val set, last epoch)
        _, val_preds_final, val_labels_final = test(model, val_loader, return_predictions=True)
        fold_acc = accuracy_score(val_labels_final, val_preds_final)
        fold_f1  = f1_score(val_labels_final,  val_preds_final, zero_division=0)
        fold_mcc = matthews_corrcoef(val_labels_final, val_preds_final)
        print(f"Fold {fold+1} — Val Acc {fold_acc:.4f} | F1 {fold_f1:.4f} | MCC {fold_mcc:.4f}")
        gnn_fold_results.append({'accuracy': fold_acc, 'f1': fold_f1, 'mcc': fold_mcc})
        

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\nGNN cross-validation results (best hyperparameters):")
    accs = [d['accuracy'] for d in gnn_fold_results]
    f1s  = [d['f1']       for d in gnn_fold_results]
    mccs = [d['mcc']      for d in gnn_fold_results]
    print(f"  Per-fold Acc: {accs}")
    print(f"  Per-fold F1:  {f1s}")
    print(f"  Per-fold MCC: {mccs}")
    print(f"  Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Mean MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"\nBest Optuna hyperparameters used:")
    for k, v in best_hp.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()