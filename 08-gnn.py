import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
import csv
from itertools import combinations
import json, os
from imblearn.under_sampling import CondensedNearestNeighbour

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

#chyba niepotrzebny:
#import seaborn as sns

def read_sequences(path, max_len):
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

        # Try literal_eval for valid python list-like strings
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                arr = np.array(parsed, dtype=float)
                return arr.flatten() if arr.ndim > 1 else arr
        except Exception:
            pass

        # Regex extract all floats in the string
        try:
            nums = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', s)]
            if len(nums) == 3:
                return np.array(nums, dtype=float)
            elif len(nums) == 6:
                arr = np.array(nums, dtype=float).reshape(2,3)
                # Choose how to handle this 2x3 — here we pick first row:
                return arr[0]
            else:
                # if unexpected number of floats, return original string to notice problem
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

def count_euclid_dist(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def count_planar_angle_piel(p1, p2, p3):
    for idx, p in enumerate([p1, p2, p3], start=1):
        print(f"p{idx}: type={type(p)}, value={p}")
    return np.dot(p2 - p1, p3 - p1)  # Example logic














#grafy
def get_graph(row, cols):
        edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [], 
                      (1,2): [], (1,3): [], (1,4): [], (1,5): [], 
                      (2,3): [], (2,4): [], (2,5): [], 
                      (3,4): [], (3,5): [], 
                      (4,5): []}

        for i in range(len(cols)):
            weigth = row[i]
            #if len(cols[i]) == 4:
                # edges_dict[(cols[i][0], cols[i][1])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][3])].append(weigth)
                #edges_dict[(cols[i][1], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][1], cols[i][3])].append(weigth)
                # edges_dict[(cols[i][2], cols[i][3])].append(weigth)
            # elif len(cols[i]) == 3:
            #     edges_dict[(cols[i][0], cols[i][1])].append(weigth)
            #     edges_dict[(cols[i][0], cols[i][2])].append(weigth)
            #     edges_dict[(cols[i][1], cols[i][2])].append(weigth)
            #else:
            if len(cols[i])==2:
                edges_dict[(cols[i][0], cols[i][1])].append(weigth)

        edge_attr = []
        for k, v in edges_dict.items():
            edge_attr.append(v)
            print(f'{k},{len(v)}')
        edge_attr =torch.tensor(edge_attr,dtype=torch.float32)

        edge_index_list = torch.tensor([edge for edge in edges_dict.keys()], dtype=torch.int64).t().contiguous()
        #edge_weights = torch.tensor([w for w in edges_dict.values()], dtype=torch.float32)
        
        #JP
        # edge_weights = []
        # #JP
        # #edges_w = torch.tensor([w for w in edges_dict.values()], dtype=torch.float32)
        # edges_w =np.array([w for w in edges_dict.values()])
        # for x in edges_w:
        #     pca = PCA(n_components=1)
        #     pca.fit(x.reshape(-1, 1))
        #     edge_weights.append(pca.singular_values_*400)
        # edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        #make graph undirected
        edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)
        #JP edge_weights_symmetric = torch.cat([edge_weights, edge_weights])

        d = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
        y = torch.tensor([row['class']], dtype=torch.int64)
        x = torch.tensor([[d[n]] for n in row['seq']], dtype=torch.float32)
        # graph = Data(edge_index=edge_index_symmetric,edge_attr=edge_attr, y=y, x=x)
        graph = Data(edge_index=edge_index_list,edge_attr=edge_attr, y=y, x=x)#edge_weight=edge_weights_symmetric

        return graph
def get_graph_hot_encoding(row, cols):
        edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [], 
                      (1,2): [], (1,3): [], (1,4): [], (1,5): [], 
                      (2,3): [], (2,4): [], (2,5): [], 
                      (3,4): [], (3,5): [], 
                      (4,5): []}

        for i in range(len(cols)):
            weigth = row[i]
            #if len(cols[i]) == 4:
                # edges_dict[(cols[i][0], cols[i][1])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][3])].append(weigth)
                #edges_dict[(cols[i][1], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][1], cols[i][3])].append(weigth)
                # edges_dict[(cols[i][2], cols[i][3])].append(weigth)
            # elif len(cols[i]) == 3:
            #     edges_dict[(cols[i][0], cols[i][1])].append(weigth)
            #     edges_dict[(cols[i][0], cols[i][2])].append(weigth)
            #     edges_dict[(cols[i][1], cols[i][2])].append(weigth)
            #else:
            if len(cols[i])==2:
                edges_dict[(cols[i][0], cols[i][1])].append(weigth)

        edge_attr = []
        for k, v in edges_dict.items():
            edge_attr.append(v)
            print(f'{k},{len(v)}')
        edge_attr =torch.tensor(edge_attr,dtype=torch.float32)

        edge_index_list = torch.tensor([edge for edge in edges_dict.keys()], dtype=torch.int64).t().contiguous()
        #edge_weights = torch.tensor([w for w in edges_dict.values()], dtype=torch.float32)
        #make graph undirected
        edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)
        #JP edge_weights_symmetric = torch.cat([edge_weights, edge_weights])
        # One-hot encoding for node features
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
    #string might contain any number of single digit numbers following one another, e.g ts125 this function will return [1,2,5]
    return [int(char) for char in string if char.isdigit()]


#TODO: upewnić się że ta funkcja jest kompatybilna z nowymi danymi
def get_graph_hot_encoding_continuity(row, cols):
        edges_dict = {(0,1): [], (0,2): [], (0,3): [], (0,4): [], (0,5): [], 
                      (1,2): [], (1,3): [], (1,4): [], (1,5): [], 
                      (2,3): [], (2,4): [], (2,5): [], 
                      (3,4): [], (3,5): [], 
                      (4,5): []}

        for i in range(len(cols)):
            weigth = row[i]
            cols_indexes = get_all_indexes_from_string(cols[i])
            if len(cols_indexes) == 2 and 7 not in cols_indexes: #ignore any indexes that are 7 (8th nucleotide, as we only want nucleotides from 1 to 6)
                #print(cols_indexes)
                edges_dict[(cols_indexes[0]-1, cols_indexes[1]-1)].append(weigth)
            #if len(cols[i]) == 4:
                # edges_dict[(cols[i][0], cols[i][1])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][0], cols[i][3])].append(weigth)
                #edges_dict[(cols[i][1], cols[i][2])].append(weigth)
                # edges_dict[(cols[i][1], cols[i][3])].append(weigth)
                # edges_dict[(cols[i][2], cols[i][3])].append(weigth)
            # elif len(cols[i]) == 3:
            #     edges_dict[(cols[i][0], cols[i][1])].append(weigth)
            #     edges_dict[(cols[i][0], cols[i][2])].append(weigth)
            #     edges_dict[(cols[i][1], cols[i][2])].append(weigth)
            #else:
            # if len(cols[i])==2:
            #     edges_dict[(cols[i][0], cols[i][1])].append(weigth)
           

        edge_attr = []
        edge_index_list = []
        
        for (i, j), weights in edges_dict.items():
            if len(weights) == 0:
                continue
            is_consecutive = 1.0 if abs(i - j) == 1 else 0.0
            is_consecutive_vec = {1,0}
            if is_consecutive == 0.0:
                is_consecutive_vec = {0,1}
            avg_weight = sum(weights) / len(weights)
            edge_attr.append(weights+[is_consecutive]) #[avg_weight, is_consecutive],
            edge_index_list.append([i, j])

        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        edge_index_list = torch.tensor(edge_index_list, dtype=torch.int64).t().contiguous()

        # Make graph undirected
        edge_index_symmetric = torch.cat([edge_index_list, edge_index_list.flip(0)], dim=1)
        edge_attr_symmetric = torch.cat([edge_attr, edge_attr], dim=0)

        # One-hot encode node features
        d = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        num_classes = len(d)
        x = torch.nn.functional.one_hot(
            torch.tensor([d[n] for n in row['seq']], dtype=torch.long), 
            num_classes=num_classes
        ).to(torch.float32)
        
        y = torch.tensor([row['is_positive']], dtype=torch.int64)

        graph = Data(edge_index=edge_index_symmetric, edge_attr=edge_attr_symmetric, y=y, x=x)
        return graph

def display_graph_and_weights(graph):
    print("Edge Index:")
    print(graph.edge_index)
    print("\nEdge Attributes (weights):")
    print(graph.edge_attr)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        # self.conv1 = GCNConv(1, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, 2) #hidden channels
        
        # self.conv1 = GCNConv(4, 64)
        # self.conv2 = GCNConv(64, 128)
        # self.conv3 = GCNConv(128, 2) #hidden channels
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 2)
        #self.soft = Softmax()

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        #x = self.soft(x)
        
        return x
def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.  Make France great again
         optimizer.zero_grad()  # Clear gradients.
         #edge_weight = data.edge_attr[:, 0]  # Only use weights, ignore is_consecutive_flag
         edge_weight = data.edge_attr[:, 0]
         edge_weight = torch.abs(edge_weight)
         
         out = model(data.x, data.edge_index, edge_weight, data.batch)  # Perform a single forward pass.
         #out= model(data.x, data.edge_index, data.edge_weight, data.batch)
        #  print(out)
        #  print("??????????????????????????????")
        #  print(data.x)
        #  print(data.edge_index)
        #  print(data.edge_weight)
        #  print(data.y)
        #  print("??????????????????????????????")
         loss = criterion(out, data.y)  # Compute the loss.
        # print(loss)
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         

def test(loader, return_predictions=False):
     model.eval()

     correct = 0
     all_preds = []
     all_labels = []
     
     with torch.no_grad():
         for data in loader:  # Iterate in batches over the training/test dataset.
             #edge_weight = data.edge_attr[:, 0]  # Only use weights
             edge_weight = data.edge_attr[:, 0]
             edge_weight = torch.abs(edge_weight)
             
             out = model(data.x, data.edge_index, edge_weight, data.batch)  #data.edge_weight
             pred = out.argmax(dim=1)  # Use the class with highest probability.
             correct += int((pred == data.y).sum())  # Check against ground-truth labels.
             
             if return_predictions:
                 all_preds.extend(pred.cpu().numpy().tolist())
                 all_labels.extend(data.y.cpu().numpy().tolist())
     
     accuracy = correct / len(loader.dataset)
     
     if return_predictions:
         return accuracy, np.array(all_preds), np.array(all_labels)
     
     return accuracy

def plot_model_metrics_during_training(epoch_data, model_name, fold_number=None, save_path=None):
    """
    Plot accuracy, F1, and MCC metrics during model training.
    
    Parameters:
    -----------
    epoch_data : dict
        Dictionary with keys 'epochs', 'train_acc', 'test_acc', 'train_f1', 'test_f1', 
        'train_mcc', 'test_mcc' containing lists of values for each epoch
    model_name : str
        Name of the model (e.g., 'GNN', 'SVM', 'GaussianNB')
    fold_number : int, optional
        Fold number for title/filename
    save_path : str, optional
        Path to save the figure. If None, won't save.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'{model_name} Metrics During Training{f" - Fold {fold_number}" if fold_number else ""}',
        fontsize=14, fontweight='bold'
    )
    
    epochs = epoch_data.get('epochs', [])
    
    # Plot 1: Accuracy
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
    
    # Plot 2: F1 Score
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
    
    # Plot 3: MCC
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
    
    #plt.show()
    return fig










dpos = pd.read_csv("positve.csv", sep=',', index_col=None)#, index_col=0)

    



#print(dpos)

dneg = pd.read_csv("negative.csv", sep=',', index_col=None)#, index_col=0)

#print(dneg)
data_full = pd.concat([dneg, dpos])
#print(data_full)

# Reading sequences data from CSV file
import csv
seqs = []

max_len = data_full.shape[0]

read_sequences("positve_seq.csv", max_len)
read_sequences("negative_seq.csv", max_len)
#add a column to data_full with sequences
data_full['seq'] = seqs
#print("DATA FULL WITH SEQS:")
#print(data_full)

#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME




dftofilter = pd.read_csv("filtered_geometric_features.csv", sep=',', index_col=0)

num_gnra_pre_filter = dftofilter["gnra"].value_counts()
#filter the dataframe to remove the clusters of data. 



# Load clusters.json and flatten all "members" lists into one list of strings

clusters_path = "clusters.json"
all_cluster_members = []
if os.path.exists(clusters_path):
    with open(clusters_path, 'r') as f:
        clusters_data = json.load(f)
    # Normalize to a list of cluster dicts
    if isinstance(clusters_data, dict):
        # if clusters are stored under a top-level key (like 'clusters'), use it
        if 'clusters' in clusters_data and isinstance(clusters_data['clusters'], list):
            clusters_list = clusters_data['clusters']
        else:
            # otherwise assume dict values are cluster dicts
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
            # try splitting comma-separated string
            parts = [s.strip() for s in members.split(',') if s.strip()]
            all_cluster_members.extend(parts)
else:
    print(f"clusters.json not found at {clusters_path}")

print(f"Total cluster members collected: {len(all_cluster_members)}")
print(f"Sample cluster members: {all_cluster_members[:10]}")

#remove .cif from all_cluster_members strings
all_cluster_members = [m.replace('.cif', '') for m in all_cluster_members]
# remove all rows from dftofilter where the source_file (which is the filename without .cif) is in all_cluster_members
print(dftofilter)
dfpreresample = dftofilter[~dftofilter.index.isin(all_cluster_members)]
print(dfpreresample)
num_gnra_post_filter = dfpreresample["gnra"].value_counts()
#df = dftofilter 

# #resample =========================================================
# Separate features and target
iks = dfpreresample.drop(columns=['gnra'])
igrek = dfpreresample['gnra']

# Apply Condensed Nearest Neighbour
cnn = CondensedNearestNeighbour(random_state=42)

cnn.fit_resample(iks, igrek)

# Use the selected indices to slice the original dataframe
df = dfpreresample.iloc[cnn.sample_indices_]
# #end resample ======================================================

stat, p_value = shapiro(df)
print(f'Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}')

#REBALANCING THE DATASET BY REDUCING THE NUMBER OF NEGATIVE SAMPLES ('gnra' = false)
# num_positive = df[df['gnra'] == True].shape[0]
# negatives = df[df['gnra'] == False]
# negatives_downsampled = negatives.sample(n=num_positive*2, random_state=42)
# df = pd.concat([df[df['gnra'] == True], negatives_downsampled])




# data_full = data_full.reset_index(drop=True)   #base dataframe with nucleotides coords and class
# y = data_full['is_positive']  #labels


#DIVIDING DATASET BY DATES
#pre,post = filter_script.filter_pandas_dataframe_by_date(df,'rna_pdb_release_dates.csv','2024-10-20T00:00:00+0000')
#trying to see if the model just learns to say yes every time
pre,post = filter_script.filter_pandas_dataframe_by_date(df,'rna_pdb_release_dates.csv','2024-10-20T00:00:00+0000')

# keep the original row indices so we can map sequences/graphs later
pre_indices = pre.index.copy()
post_indices = post.index.copy()
print(f"number of rows in df: {df.shape[0]}")
print("============================================ DF PRE selected date============================================")
print(pre)
print("============================================ DF POST selected date ============================================")
print(post)
print("============================================ END ============================================")
num_gnra_in_post_df= post["gnra"].value_counts()
num_all_files_in_post_df = post.shape[0]
num_gnra_in_pre_df= pre["gnra"].value_counts()
num_all_files_in_pre_df = pre.shape[0]
y = df['gnra']
y_pre = pre['gnra']
y_post = post['gnra']
data_full.iloc[180]


#remove the 'gnra' column from df to get only features
df = df.drop(columns=['gnra'])
pre = pre.drop(columns=['gnra'])
post = post.drop(columns=['gnra'])

# use stratified k-fold on the pre dataset, validating each fold on the fixed post set
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nUsing {n_splits}-fold stratified cross-validation on pre, validating on post.")

# reset indexes
pre = pre.reset_index(drop=True)
post = post.reset_index(drop=True)
y_pre = y_pre.reset_index(drop=True)
y_post = y_post.reset_index(drop=True)

# container for results
cv_results = {}  # will hold evaluation returned by helper




# define evaluation helper that trains on pre and validates on post

def evaluate_classifiers(X_train, y_train, X_val, y_val, prefix=''):
    results = {}
    # GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_val)
    results['GaussianNB'] = {
        'accuracy': accuracy_score(y_val, y_pred_gnb),
        'precision': precision_score(y_val, y_pred_gnb),
        'recall': recall_score(y_val, y_pred_gnb),
        'f1': f1_score(y_val, y_pred_gnb)
    }
    gnb_metrics = {
        'epochs': [1],
        'train_acc': [accuracy_score(y_train, gnb.predict(X_train))],
        'test_acc': [results['GaussianNB']['accuracy']],
        'train_f1': [f1_score(y_train, gnb.predict(X_train), zero_division=0)],
        'test_f1': [results['GaussianNB']['f1']],
        'train_mcc': [matthews_corrcoef(y_train, gnb.predict(X_train))],
        'test_mcc': [matthews_corrcoef(y_val, y_pred_gnb)]
    }
    plot_model_metrics_during_training(gnb_metrics, 'GaussianNB', save_path=f'gnb_metrics_{prefix}.png')

    # SVM
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)
    clf = svm.SVC()
    clf.fit(X_train_scaled, y_train)
    y_pred_svm = clf.predict(X_val_scaled)
    results['SVM'] = {
        'accuracy': accuracy_score(y_val, y_pred_svm),
        'precision': precision_score(y_val, y_pred_svm),
        'recall': recall_score(y_val, y_pred_svm),
        'f1': f1_score(y_val, y_pred_svm)
    }
    svm_metrics = {
        'epochs': [1],
        'train_acc': [accuracy_score(y_train, clf.predict(X_train_scaled))],
        'test_acc': [results['SVM']['accuracy']],
        'train_f1': [f1_score(y_train, clf.predict(X_train_scaled), zero_division=0)],
        'test_f1': [results['SVM']['f1']],
        'train_mcc': [matthews_corrcoef(y_train, clf.predict(X_train_scaled))],
        'test_mcc': [matthews_corrcoef(y_val, y_pred_svm)]
    }
    plot_model_metrics_during_training(svm_metrics, 'SVM', save_path=f'svm_metrics_{prefix}.png')

    print(f"\nEvaluation results ({prefix}):")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")
    return results

# perform evaluation using defined split
cv_results = evaluate_classifiers(pre, y_pre, post, y_post, prefix='pre_post')


# Keep standardized copy for GNN
scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df))
df_std.columns = df.columns

# Build a graph-ready DataFrame (features, seq, label)
seqs_series = pd.Series(seqs, name='seq')
if len(seqs_series) > df_std.shape[0]:
    seqs_series = seqs_series.iloc[: df_std.shape[0]].reset_index(drop=True)
elif len(seqs_series) < df_std.shape[0]:
    seqs_series = pd.Series([''] * df_std.shape[0], name='seq')

# df_graph used later for GNN folds
df_graph = df_std.reset_index(drop=True).copy()
df_graph['seq'] = seqs_series
df_graph['is_positive'] = y.reset_index(drop=True).astype(int)

#GAUSSIAN BAYASES  # <- single-split block removed (using StratifiedKFold above)
# old single-split code removed (replaced with StratifiedKFold above)
# single-split fit removed; model is trained inside the StratifiedKFold loop above

# single-split prediction removed; predictions are computed in the CV loop above

# Single-split GNB evaluation removed — see cross-validation summary above

df_d = df.copy()
df_d['is_positive'] = y



# Wykresy gęstości cech dla każdej klasy
# for feature in list(df.columns):  # Pomijamy kolumnę 'label'
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=df_d, x=feature, hue='class', fill=True)
#     plt.title(f'Feature Density Plot for {feature}')
#     plt.show()

# Per-fold statistics (means/stds and single-split diagnostics removed) — use CV summaries above for performance assessment

# Single-split SVM evaluation removed — SVM results are reported above from StratifiedKFold cross-validation
# df_std remains defined above and will be used to construct graph datasets for the GNN.

# Removed single-split misclassification listing; use per-fold diagnostics if needed









# GNN — use k-fold on pre dataset, validating each fold on fixed post dataset
print("\nUsing StratifiedKFold on pre for GNN training/evaluation, with post as validation set...")

# compute positional mappings from original df index to df_graph rows
# pos_pre = df.index.get_indexer(pre_indices)
# pos_post = df.index.get_indexer(post_indices)

# construct graph-specific DataFrames
# df_graph_pre = df_graph.iloc[pos_pre].reset_index(drop=True)
# df_graph_post = df_graph.iloc[pos_post].reset_index(drop=True)
# pre and post are already the correct row subsets of df (features only, reset index)
# y_pre and y_post are the corresponding labels (reset index)

scaler_pre = StandardScaler()
pre_std = pd.DataFrame(scaler_pre.fit_transform(pre), columns=pre.columns)

scaler_post = StandardScaler()  
post_std = pd.DataFrame(scaler_post.fit_transform(post), columns=post.columns)

df_graph_pre = pre_std.copy()
df_graph_pre['seq'] = [seqs[i] for i in pre_indices] if hasattr(pre_indices, '__iter__') else [''] * len(pre_std)
df_graph_pre['is_positive'] = y_pre.values

df_graph_post = post_std.copy()
df_graph_post['seq'] = [seqs[i] for i in post_indices] if hasattr(post_indices, '__iter__') else [''] * len(post_std)
df_graph_post['is_positive'] = y_post.values

# Verify immediately
print("df_graph_pre class balance:", df_graph_pre['is_positive'].value_counts())
print("df_graph_post class balance:", df_graph_post['is_positive'].value_counts())



print(f"previously gathered data \n gnra in post: {num_gnra_in_post_df} len post: {num_all_files_in_post_df} \n gnra in post: {num_gnra_in_pre_df} len post: {num_all_files_in_pre_df}")
print("df_graph_pre class balance:")
print(df_graph_pre['is_positive'].value_counts())
print("df_graph_post class balance:")
print(df_graph_post['is_positive'].value_counts())
print(f"Total graphs in pre: {len(df_graph_pre)}")
print(f"Total graphs in post (fixed validation): {len(df_graph_post)}")

# prepare post dataset once (shared across all folds)
cols_post = df_graph_post.columns[:-1]
test_dataset = df_graph_post.apply(lambda x: get_graph_hot_encoding_continuity(x, cols_post), axis=1)
test_loader = DataLoader(test_dataset, batch_size=32)

gnn_fold_results = []

# Keep standardized copy for GNN
# scaler = StandardScaler()
# df_std = pd.DataFrame(scaler.fit_transform(df))
# df_std.columns = df.columns

# # Build a graph-ready DataFrame (features, seq, label)
# seqs_series = pd.Series(seqs, name='seq')
# if len(seqs_series) > df_std.shape[0]:
#     seqs_series = seqs_series.iloc[: df_std.shape[0]].reset_index(drop=True)
# elif len(seqs_series) < df_std.shape[0]:
#     seqs_series = pd.Series([''] * df_std.shape[0], name='seq')

# # df_graph used later for GNN folds
# df_graph = df_std.reset_index(drop=True).copy()
# df_graph['seq'] = seqs_series
# df_graph['is_positive'] = y.reset_index(drop=True).astype(int)

#GAUSSIAN BAYASES  # <- single-split block removed (using StratifiedKFold above)
# old single-split code removed (replaced with StratifiedKFold above)
# single-split fit removed; model is trained inside the StratifiedKFold loop above

# single-split prediction removed; predictions are computed in the CV loop above

# Single-split GNB evaluation removed — see cross-validation summary above

df_d = df.copy()
df_d['is_positive'] = y



# Wykresy gęstości cech dla każdej klasy
# for feature in list(df.columns):  # Pomijamy kolumnę 'label'
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=df_d, x=feature, hue='class', fill=True)
#     plt.title(f'Feature Density Plot for {feature}')
#     plt.show()

# Per-fold statistics (means/stds and single-split diagnostics removed) — use CV summaries above for performance assessment

# Single-split SVM evaluation removed — SVM results are reported above from StratifiedKFold cross-validation
# df_std remains defined above and will be used to construct graph datasets for the GNN.

# Removed single-split misclassification listing; use per-fold diagnostics if needed









# GNN — use Stratified K-Fold to train and evaluate the graph network
# We'll reuse the previously prepared `df_graph` (features standardized, with 'seq' and 'is_positive')
print("\nUsing StratifiedKFold for GNN training/evaluation...")
print("Graph DataFrame head:=======================================================")
print(df_graph)
print("Graph DataFrame columns:")
print(df_graph.columns)
#print(df_graph['gnra'])
gnn_fold_results = []#                                                        removed 'seq' from columns that are dropped
USE_FULL_DATASET = False   # Set to False to use normal k-fold CV
# ────────────────────────────────────────────────────────────────────────────

if USE_FULL_DATASET:
    # Single "fake" fold using all data as both train and val
    fold_splits = [(df_graph_pre.index.tolist(), df_graph_pre.index.tolist())]
else:
    fold_splits = list(skf.split(df_graph_pre.drop(columns=['is_positive']), df_graph_pre['is_positive']))

for fold, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"\n--- GNN Fold {fold + 1}/{n_splits} ---")

    # df_train = df_graph.iloc[train_idx].reset_index(drop=True)
    # df_test = df_graph.iloc[val_idx].reset_index(drop=True)
    df_train = df_graph_pre.iloc[train_idx].reset_index(drop=True)
    df_test  = df_graph_pre.iloc[val_idx].reset_index(drop=True)
    # After creating df_train, before initializing the model
    pos = (df_train['is_positive'] == 1).sum()
    neg = (df_train['is_positive'] == 0).sum()
    print(f"NEG: {neg} POS: {pos}")
    total = len(df_train)

    # Directly use the ratio as the positive class weight
    pos_weight_value = (neg / pos)
    print(f"Positives: {pos}, Negatives: {neg}, Pos weight: {pos_weight_value:.2f}x")

    class_weights = torch.tensor([1.0, pos_weight_value], dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    cols = df_train.columns[:-1]  # all feature columns
    train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_continuity(x, cols), axis=1)
    test_dataset = df_test.apply(lambda x: get_graph_hot_encoding_continuity(x, cols), axis=1)

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model, optimizer, loss
    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights) 
    
    # Per-fold early stopping
    best_model_state = None
    best_model_state_by_acc = None
    best_acc = 0.0
    best_f1 = 0.0
    no_improve_counter = 0
    max_no_improve = 10
    
    # Tracking metrics for plotting
    epoch_metrics = {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'train_f1': [],
        'test_f1': [],
        'train_mcc': [],
        'test_mcc': []
    }

    max_epochs = 200
    for epoch in range(1, max_epochs + 1):
        train()
        
        # Get predictions to calculate F1 and MCC
        train_acc, train_preds, train_labels = test(train_loader, return_predictions=True)
        test_acc, test_preds, test_labels = test(test_loader, return_predictions=True)
        
        # Calculate F1 and MCC
        if epoch == 40:
            print(f"=======================================================================\n {train_labels} \n{train_preds}")
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, zero_division=0)
        train_mcc = matthews_corrcoef(train_labels, train_preds)
        test_mcc = matthews_corrcoef(test_labels, test_preds)
        
        # Store metrics
        epoch_metrics['epochs'].append(epoch)
        epoch_metrics['train_acc'].append(train_acc)
        epoch_metrics['test_acc'].append(test_acc)
        epoch_metrics['train_f1'].append(train_f1)
        epoch_metrics['test_f1'].append(test_f1)
        epoch_metrics['train_mcc'].append(train_mcc)
        epoch_metrics['test_mcc'].append(test_mcc)

        if test_f1 > best_f1:  # or test_mcc > best_mcc
            best_f1 = test_f1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_counter = 0
        else:
            no_improve_counter += 1
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state_by_acc = copy.deepcopy(model.state_dict())
        #     no_improve_counter = 0
        # else:
        #     no_improve_counter += 1

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f}')

        if no_improve_counter >= max_no_improve and epoch>190:
            print("Early stopping due to no improvement")
            break

    # Restore best model for this fold (for reporting)
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Plot metrics during training for this fold
    plot_model_metrics_during_training(epoch_metrics, 'GNN', fold_number=fold+1, save_path=f'gnn_training_metrics_fold_{fold+1}.png')

    # Evaluate restored model on test set to collect predictions and compute metrics
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            edge_weight = data.edge_attr[:, 0]
            out = model(data.x, data.edge_index, edge_weight, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    # Diagnostic outputs to detect collapsed predictions / class imbalance
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
    fold_f1 = f1_score(y_true, y_pred, zero_division=0)
    fold_mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Fold {fold + 1} best Test Acc: {best_acc:.4f}")
    print(f"Fold {fold + 1} metrics: Acc={fold_acc:.4f}, F1={fold_f1:.4f}, MCC={fold_mcc:.4f}")
    gnn_fold_results.append({'accuracy': fold_acc, 'f1': fold_f1, 'mcc': fold_mcc})
# print("------------------------------------------------------")
# display_graph_and_weights(train_dataset.iloc[0])  # Display the first graph's edge index and attributes for verification
# print("------------------------------------------------------")
# display_graph_and_weights(test_dataset.iloc[0])
# print("------------------------------------------------------")
# Final GNN CV summary
print("\nGNN cross-validation results:")
accs = [d['accuracy'] for d in gnn_fold_results]
f1s = [d['f1'] for d in gnn_fold_results]
mccs = [d['mcc'] for d in gnn_fold_results]
print(f"  Per-fold accuracy: {accs}")
print(f"  Per-fold F1: {f1s}")
print(f"  Per-fold MCC: {mccs}")
print(f"  Mean accuracy: {np.mean(accs):.4f} (std: {np.std(accs):.4f})")
print(f"  Mean F1: {np.mean(f1s):.4f} (std: {np.std(f1s):.4f})")
print(f"  Mean MCC: {np.mean(mccs):.4f} (std: {np.std(mccs):.4f})")
# print(cv_results["GaussianNB"]["accuracy"])
# print(cv_results["GaussianNB"]["recall"])
# print(cv_results["GaussianNB"]["precision"])
# print(cv_results["GaussianNB"]["f1"])
#TODO remove redundancy 









#!!!!!!!!!!!!!!!!!!!!!!!!!!!SKF VERSION WHICH HAD 0 ACCURACY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #for fold, (train_idx, val_idx) in enumerate(skf.split(df_graph_pre.drop(columns=['is_positive']), df_graph_pre['is_positive'])):
# # ── SPLIT OVERRIDE ──────────────────────────────────────────────────────────
# USE_FULL_DATASET = True   # Set to False to use normal k-fold CV
# # ────────────────────────────────────────────────────────────────────────────

# if USE_FULL_DATASET:
#     # Single "fake" fold using all data as both train and val
#     fold_splits = [(df_graph_pre.index.tolist(), df_graph_pre.index.tolist())]
# else:
#     fold_splits = list(skf.split(df_graph_pre.drop(columns=['is_positive']), df_graph_pre['is_positive']))

# for fold, (train_idx, val_idx) in enumerate(fold_splits):
#     print(f"\n--- GNN Fold {fold + 1}/{n_splits} ---")

#     df_train = df_graph_pre.iloc[train_idx].reset_index(drop=True)
#     df_val = df_graph_pre.iloc[val_idx].reset_index(drop=True)

#     cols_train = df_train.columns[:-1]
#     train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_continuity(x, cols_train), axis=1)
#     val_dataset = df_val.apply(lambda x: get_graph_hot_encoding_continuity(x, cols_train), axis=1)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32)

#     print(f'Training graphs: {len(train_dataset)}, Validation (pre) graphs: {len(val_dataset)}, Test (post) graphs: {len(test_dataset)}')

#     # Initialize model, optimizer, loss
#     model = GCN(hidden_channels=64)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = torch.nn.CrossEntropyLoss()

#     # Early stopping
#     best_model_state = None
#     best_acc = 0.0
#     no_improve_counter = 0
#     max_no_improve = 10
    
#     # Tracking metrics for plotting
#     epoch_metrics = {
#         'epochs': [],
#         'train_acc': [],
#         'test_acc': [],
#         'train_f1': [],
#         'test_f1': [],
#         'train_mcc': [],
#         'test_mcc': []
#     }

#     max_epochs = 200
#     for epoch in range(1, max_epochs + 1):
#         train()
        
#         # Get predictions to calculate F1 and MCC
#         train_acc, train_preds, train_labels = test(train_loader, return_predictions=True)
#         test_acc, test_preds, test_labels = test(test_loader, return_predictions=True)
        
#         # Calculate F1 and MCC
#         train_f1 = f1_score(train_labels, train_preds, zero_division=0)
#         test_f1 = f1_score(test_labels, test_preds, zero_division=0)
#         train_mcc = matthews_corrcoef(train_labels, train_preds)
#         test_mcc = matthews_corrcoef(test_labels, test_preds)
        
#         # Store metrics
#         epoch_metrics['epochs'].append(epoch)
#         epoch_metrics['train_acc'].append(train_acc)
#         epoch_metrics['test_acc'].append(test_acc)
#         epoch_metrics['train_f1'].append(train_f1)
#         epoch_metrics['test_f1'].append(test_f1)
#         epoch_metrics['train_mcc'].append(train_mcc)
#         epoch_metrics['test_mcc'].append(test_mcc)

#         if test_acc > best_acc:
#             best_acc = test_acc
#             best_model_state = copy.deepcopy(model.state_dict())
#             no_improve_counter = 0
#         else:
#             no_improve_counter += 1

#         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f}')

#         if no_improve_counter >= max_no_improve and epoch > 50:
#             print("Early stopping due to no improvement")
#             break

#     # Restore best model for this fold
#     if best_model_state:
#         model.load_state_dict(best_model_state)

#     # Plot metrics during training for this fold
#     plot_model_metrics_during_training(epoch_metrics, 'GNN', fold_number=fold+1, save_path=f'gnn_training_metrics_fold_{fold+1}.png')

#     # Evaluate on post set
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for data in test_loader:
#             edge_weight = data.edge_attr[:, 0]
#             out = model(data.x, data.edge_index, edge_weight, data.batch)
#             pred = out.argmax(dim=1)
#             y_true.extend(data.y.cpu().numpy().tolist())
#             y_pred.extend(pred.cpu().numpy().tolist())

#     print('y_true dist:', np.bincount(y_true) if len(y_true) > 0 else 'empty')
#     print('y_pred dist:', np.bincount(y_pred) if len(y_pred) > 0 else 'empty')
#     print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
#     print('classification report:\n', classification_report(y_true, y_pred, zero_division=0))

#     fold_acc = accuracy_score(y_true, y_pred)
#     fold_f1 = f1_score(y_true, y_pred, zero_division=0)
#     fold_mcc = matthews_corrcoef(y_true, y_pred)
#     print(f"Fold {fold + 1} best Test Acc: {best_acc:.4f}")
#     print(f"Fold {fold + 1} metrics on post: Acc={fold_acc:.4f}, F1={fold_f1:.4f}, MCC={fold_mcc:.4f}")
#     gnn_fold_results.append({'accuracy': fold_acc, 'f1': fold_f1, 'mcc': fold_mcc})

# # Final GNN results summary
# print("\nGNN cross-validation results (evaluated on post set):")
# accs = [d['accuracy'] for d in gnn_fold_results]
# f1s = [d['f1'] for d in gnn_fold_results]
# mccs = [d['mcc'] for d in gnn_fold_results]
# print(f"  Per-fold accuracy: {accs}")
# print(f"  Per-fold F1: {f1s}")
# print(f"  Per-fold MCC: {mccs}")
# print(f"  Mean accuracy: {np.mean(accs):.4f} (std: {np.std(accs):.4f})")
# print(f"  Mean F1: {np.mean(f1s):.4f} (std: {np.std(f1s):.4f})")
# print(f"  Mean MCC: {np.mean(mccs):.4f} (std: {np.std(mccs):.4f})")

# print('post class distribution:', np.bincount(y_post))
# print('post class distribution:', np.bincount(y_pre))

# print("distribution pre first filter")
# print(num_gnra_pre_filter)
# print("distribution post first filter")
# print(num_gnra_post_filter)
# print("distribution dates pre")
# print(num_gnra_in_pre_df)
# print("distribution dates post")
# print(num_gnra_in_post_df)