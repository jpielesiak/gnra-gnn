import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
import csv
from itertools import combinations


from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
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

        y = torch.tensor([row['class']], dtype=torch.int64)
        graph = Data(edge_index=edge_index_list, edge_attr=edge_attr, y=y, x=x)

        return graph


def get_graph_hot_encoding_continuity(row, cols):
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
        edge_index_list = []
        
        for (i, j), weights in edges_dict.items():
            if len(weights) == 0:
                continue
            is_consecutive = 1.0 if abs(i - j) == 1 else 0.0
            avg_weight = sum(weights) / len(weights)
            edge_attr.append([avg_weight, is_consecutive])
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

        y = torch.tensor([row['class']], dtype=torch.int64)

        graph = Data(edge_index=edge_index_symmetric, edge_attr=edge_attr_symmetric, y=y, x=x)
        return graph
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        # self.conv1 = GCNConv(1, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, 2) #hidden channels
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 2) #hidden channels
        self.soft = Softmax()

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
        x = self.soft(x)
        
        return x
def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.  Make France great again
         edge_weight = data.edge_attr[:, 0]  # Only use weights, ignore is_consecutive_flag
         out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
        #  print(out)
        #  print(data.y)
         loss = criterion(out, data.y)  # Compute the loss.
        # print(loss)
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         edge_weight = data.edge_attr[:, 0]  # Only use weights
         out = model(data.x, data.edge_index, data.edge_weight, data.batch)  
         #print(out)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         #print(pred)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

















# #MAIN
# data = pd.read_csv("./database/database_C1'.csv", sep=',', index_col=0)
# data= data.map(parse_point)
# data['class'] = 1
# data

# #stems
# d1 = pd.read_csv("./stems_database_C1'.csv", sep=',', index_col=0)
# # for column in d1.columns[:-1]:
# #     if d1[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #         d1[column] = pd.to_numeric(d1[column])
#     #d1[column] = d1[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d1=d1.map(parse_point)
# d1 = d1.reset_index(drop=True)

# #hairpins
# d2 = pd.read_csv("./hairpins_database_C1'.csv", index_col=0, sep=',')
# # for column in d2.columns[:-1]:
# #     if d2[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #         d2[column] = pd.to_numeric(d2[column])
# #     d2[column] = d2[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d2=d2.map(parse_point)
# d2 = d2.reset_index(drop=True)
# d2

# #internal loops
# d3 = pd.read_csv("./loops_database_C1'.csv", sep=',', index_col=0)
# # for column in d3.columns[:-1]:
# #     if d3[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #         d3[column] = pd.to_numeric(d3[column])
#     #d3[column] = d3[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d3=d3.map(parse_point)
# d3 = d3.reset_index(drop=True)
# d3

# #ends
# d4 = pd.read_csv("./ends_database_C1'.csv", sep=',', index_col=0)
# # for column in d4.columns[:-1]:
# #     if d4[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #         d4[column] = pd.to_numeric(d4[column])
# #     d4[column] = d4[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d4=d4.map(parse_point)
# d4 = d4.reset_index(drop=True)
# d4

# #mix
# d5 = pd.read_csv("./mix_database_C1'.csv", sep=',', index_col=0)
# # for column in d5.columns[:-1]:
# #     if d5[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #         d5[column] = pd.to_numeric(d5[column])
#     #d5[column] = d5[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d5=d5.map(parse_point)
# d5 = d5.reset_index(drop=True)
# d5

# #mix_with_seqs_assigned
# d5v2 = pd.read_csv("./mix_database_C1_v2'.csv", sep=',', index_col=0)
# # for column in d5v2.columns[:-1]:
# #     if d5v2[column].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
# #        d5v2[column] = pd.to_numeric(d5v2[column])
#     #d5v2[column] = d5v2[column].apply(lambda x: np.array(list(map(float, x.split(',')))))
# d5v2=d5v2.map(parse_point)
# d5v2 = d5v2.reset_index(drop=True)
# d5v2
dpos = pd.read_csv("positve.csv", sep=',', index_col=0)
dneg = pd.read_csv("negative.csv", sep=',', index_col=0)
data_full = pd.concat([dneg, dpos])
print(data_full)

# Reading sequences data from CSV file
import csv
seqs = []

# with open("./database/database_C1'_seqs.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         seqs.append(row[0])

# with open("./stems_database_C1'_seqs.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         seqs.append(row[0])

# with open("./hairpins_database_C1'_seqs.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         seqs.append(row[0])

# with open("./loops_database_C1'_seqs.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         seqs.append(row[0])

# with open("./ends_database_C1'_seqs.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         seqs.append(row[0])

# with open("./mix_database_C1'_seqs_v2.csv", 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         if len(seqs) == data_full.shape[0]:
#             break
#         seqs.append(row[0])
with open("positve_seq.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(seqs) == data_full.shape[0]:
            break
        seqs.append(row[0])
with open("negative_seq.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(seqs) == data_full.shape[0]:
            break
        seqs.append(row[0])        





planar_angles_full = []
cols =[]
def is_bad(cell):
    return isinstance(cell, str) and (',' in cell or cell.startswith('['))

bad_cells = data_full.map(is_bad)
print("Bad cells:\n", bad_cells.any())  # Shows which columns have bad entries
print(data_full.map(lambda x: x.shape if isinstance(x, np.ndarray) else None))

for row in data_full.iloc[:, :-1].values:
    tmp = []
    for i in range(len(row) - 2):
        for j in range(i + 1, len(row) - 1):
            for k in range(j + 1, len(row)):
                print(f'{row[i]} ]]] {row[j]} ]]] {row[k]} {i},{j},{k}')
                tmp.append(count_planar_angle_piel(row[i], row[j], row[k]))
                if len(cols) < 20:
                    cols.append((i, j, k))
    planar_angles_full.append(tmp)
# for row in data_full.iloc[:,:-1].iterrows():
#     tmp = []
#     for i in range(len(row[1]) - 2):
#         for j in range(i + 1, len(row[1]) - 1):
#             for k in range(j + 1, len(row[1])):
#                 print(f'{row[1][i]} ]]] {row[1][j]} ]]] {row[1][k]} {i} ,{j},{k}')
#                 tmp.append(count_planar_angle(row[1][i], row[1][j], row[1][k]))
#                 if len(cols) < 20:
#                     cols.append((i,j,k))
#     planar_angles_full.append(tmp)

planar_angles_full = pd.DataFrame(planar_angles_full)
planar_angles_full.columns=cols
print(planar_angles_full)


torsion_angles = []
cols=[]
for row in data_full.iloc[:,:-1].iterrows():
    tmp = []
    for i in range(len(row[1])-3):  
        for j in range(i+1, len(row[1])-2):  
            for k in range(j+1, len(row[1])-1):
                for l in range(k+1, len(row[1])):
                    tmp.append(count_torsion_angle(row[1][i], row[1][j], row[1][k], row[1][l]))
                    if len(cols) < 15:
                        cols.append((i,j,k,l))
    # for combo in combinations(row[1], 4):
    #     tmp.append(count_torsion_angle(combo[0], combo[1], combo[2], combo[3]))
    torsion_angles.append(tmp)

torsion_angles_full = pd.DataFrame(torsion_angles)
torsion_angles_full.columns=cols

### SKALOWANIE ZEBY NIE BYLO UJEMNYCH ###
for i in range(torsion_angles_full.shape[0]):
    torsion_angles_full.iloc[i] = torsion_angles_full.iloc[i].apply(lambda x: x+360 if x < 0 else x)
    
print(torsion_angles_full)


distances_full = []
for row in data_full.iloc[:,:-1].iterrows():
    tmp = []
    for j in range(0,len(row[1])):
        for k in range(j+1, len(row[1])):
            tmp.append(count_euclid_dist(row[1][j], row[1][k]))
    distances_full.append(tmp)

distances_full = pd.DataFrame(distances_full)
distances_full.columns = [(j,k) for j in range(0, data.shape[1]-1) for k in range(j+1, data.shape[1]-1)]
print(distances_full)

#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
#CREATING THE DATAFRAME
df = pd.concat([planar_angles_full, torsion_angles_full, distances_full], axis=1)






stat, p_value = shapiro(df)
print(f'Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}')


data_full = data_full.reset_index(drop=True)   #base dataframe with nucleotides coords and class
y = data_full['class']

data_full.iloc[180]



#GAUSSIAN BAYASES
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=12)

gnb = GaussianNB()
model_GNB = gnb.fit(X_train, y_train)

y_pred_GNB = model_GNB.predict(X_test)

confusion_GNB = confusion_matrix(y_test, y_pred_GNB)
print(confusion_GNB)
print(("Accuracy is"), accuracy_score(y_test, y_pred_GNB) * 100)
print(("Accuracy is"), accuracy_score(y_test, y_pred_GNB) * 100)
print(("Presisions is"), precision_score(y_test, y_pred_GNB) * 100)
print(("Recall is"), recall_score(y_test, y_pred_GNB) * 100)
print(("F1 is"), f1_score(y_test, y_pred_GNB) * 100)

df_d = df.copy()
df_d['class'] = y



# Wykresy gęstości cech dla każdej klasy
# for feature in list(df.columns):  # Pomijamy kolumnę 'label'
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(data=df_d, x=feature, hue='class', fill=True)
#     plt.title(f'Feature Density Plot for {feature}')
#     plt.show()

means = model_GNB.theta_
stds = np.sqrt(model_GNB.var_)

# DataFrame ze średnimi cech dla każdej klasy
df_means = pd.DataFrame(means.T, columns=[f'Class {i}' for i in range(means.shape[0])], index=df.columns)
print("Means of features for each class:")
print(df_means)

# DataFrame z odchyleniami standardowymi cech dla każdej klasy
df_stds = pd.DataFrame(stds.T, columns=[f'Class {i}' for i in range(stds.shape[0])], index=df.columns)
print("Standard deviations of features for each class:")
print(df_stds)
# Find indices of misclassified instances

misclassified_indices = (y_test != y_pred_GNB)
misclassified_indices = misclassified_indices[misclassified_indices].index
misclassified_true_labels = y_test[misclassified_indices]

print(f"Indices wrongly assigned and their correct classes {misclassified_true_labels}")

#STANDARYZACJA

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df))
df_std.columns=df.columns

X_train, X_test, y_train, y_test = train_test_split(df_std, y, test_size=0.2, random_state=12)   #lepszy wynik po standaryzacji

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred_SVM = clf.predict(X_test)

confusion_SVM = confusion_matrix(y_test, y_pred_SVM)
print(confusion_SVM)
print(("Accuracy is"), accuracy_score(y_test, y_pred_SVM) * 100)

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the SVM classifier on the PCA-transformed data
clf = svm.SVC()  
clf.fit(X_train_pca, y_train)

# Predict on the PCA-transformed test set
y_pred_SVM = clf.predict(X_test_pca)

# Confusion matrix and accuracy
confusion_SVM = confusion_matrix(y_test, y_pred_SVM)
print(confusion_SVM)
print("Accuracy is", accuracy_score(y_test, y_pred_SVM) * 100)

misclassified_indices = (y_test != y_pred_SVM)
misclassified_indices = misclassified_indices[misclassified_indices].index
misclassified_true_labels = y_test[misclassified_indices]

print(f"Indices wrongly assigned and their correct classes {misclassified_true_labels}")









#GNN
df_std_nucs = df_std.copy()
df_std_nucs['seq'] = seqs
df_nucs = df.copy()
df_nucs['seq'] = seqs
X_train, X_test, y_train, y_test = train_test_split(df_std_nucs, y, test_size=0.2, random_state=12)
df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

cols = df_train.columns[:-2]
train_dataset = df_train.apply(lambda x: get_graph_hot_encoding_continuity(x, cols), axis=1)
test_dataset = df_test.apply(lambda x: get_graph_hot_encoding_continuity(x, cols), axis=1)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

data = train_dataset[0]  # Get the first graph object.

print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

print('=============================================================')
print(data.x)
print(data.y)
print(data.edge_index)
print(data.edge_attr)
#print(data.edge_weight)

#można regulować rozmiar batcha
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

best_model_state = None
best_acc = 0.0
no_improve_counter = 0
max_no_improve = 10
acc_history = []

for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    # Save current model state if it improves
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_state = copy.deepcopy(model.state_dict())
        no_improve_counter = 0
    else:
        no_improve_counter += 1

    acc_history.append(test_acc)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Early stopping if last 10 test acc values are decreasing
    if len(acc_history) >= 30:
        recent_accs = acc_history[-10:]
        # if all(x >= y for x, y in zip(recent_accs, recent_accs[1:])):
            # print("🔁 Accuracy has been decreasing for 10 epochs — early stopping.")
            # break

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"✅ Reverted to best model with Test Acc: {best_acc:.4f}")