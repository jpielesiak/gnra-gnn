#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations
import json
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

# Load the geometric features dataset
df = pd.read_csv("geometric_features.csv")

print(f"Original dataset shape: {df.shape}")

# Load clusters.json to get representative structures
with open("clusters.json", "r") as f:
    clusters_data = json.load(f)

# Extract representative filenames and remove .cif extension
representative_files = set()
for cluster in clusters_data["clusters"]:
    rep_file = cluster["representative"]
    if rep_file.endswith(".cif"):
        rep_file = rep_file[:-4]  # Remove .cif extension
    representative_files.add(rep_file)

print(f"Found {len(representative_files)} representative structures")

# Filter positive dataset to only contain representatives
# Only filter positive samples (gnra == 1), keep all negative samples
positive_mask = df["gnra"] == 1
negative_mask = df["gnra"] == 0

# For positive samples, keep only representatives
positive_df = df[positive_mask]
representative_mask = positive_df["source_file"].isin(representative_files)
filtered_positive_df = positive_df[representative_mask]

# Keep all negative samples
negative_df = df[negative_mask]

# Combine filtered positive and all negative samples
df = pd.concat([filtered_positive_df, negative_df], ignore_index=True)

print(f"After filtering to representatives:")
print(f"  Positive samples: {len(filtered_positive_df)} (was {len(positive_df)})")
print(f"  Negative samples: {len(negative_df)}")
print(f"  Total samples: {len(df)}")

# Generate column names to drop (raw angle values)
columns_to_drop = []

# Drop planar angle columns a{i}{j}{k}
for i, j, k in combinations(range(8), 3):
    columns_to_drop.append(f"a{i}{j}{k}")

# Drop torsion angle columns t{i}{j}{k}{l}
for i, j, k, l in combinations(range(8), 4):
    columns_to_drop.append(f"t{i}{j}{k}{l}")

# Drop the columns
df = df.drop(columns=columns_to_drop)

print(f"Dropped {len(columns_to_drop)} angle columns")
print(f"New dataset shape: {df.shape}")
print(f"Remaining columns: {list(df.columns)}")
print("GNRA distribution:")
print(df["gnra"].value_counts())

# Prepare features and target
X = df.drop(columns=["source_file", "gnra"])
y = df["gnra"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# K-fold cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nUsing {n_splits}-fold stratified cross-validation.")

# Initialize classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42),
}


# Create neural network model
def create_neural_network(input_dim):
    model = keras.Sequential(
        [
            keras.layers.Dense(50, activation="relu", input_shape=(input_dim,)),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train and evaluate each classifier using k-fold cross-validation
cv_results = {}
classifier_names = list(classifiers.keys()) + ["Neural Network"]
for name in classifier_names:
    cv_results[name] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize features for this fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate classical classifiers
    for name, classifier in classifiers.items():
        print(f"  Training and evaluating {name}...")
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)

        cv_results[name]["accuracy"].append(accuracy_score(y_test, y_pred))
        cv_results[name]["precision"].append(precision_score(y_test, y_pred))
        cv_results[name]["recall"].append(recall_score(y_test, y_pred))
        cv_results[name]["f1"].append(f1_score(y_test, y_pred))

    # Train and evaluate neural network
    print("  Training and evaluating Neural Network...")
    nn_model = create_neural_network(X_train_scaled.shape[1])
    nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred_proba = nn_model.predict(X_test_scaled, verbose=0)
    y_pred_nn = (y_pred_proba > 0.5).astype(int).flatten()

    cv_results["Neural Network"]["accuracy"].append(accuracy_score(y_test, y_pred_nn))
    cv_results["Neural Network"]["precision"].append(precision_score(y_test, y_pred_nn))
    cv_results["Neural Network"]["recall"].append(recall_score(y_test, y_pred_nn))
    cv_results["Neural Network"]["f1"].append(f1_score(y_test, y_pred_nn))

# Average results for final summary
results = {}
for name, metrics in cv_results.items():
    results[name] = {
        "accuracy": np.mean(metrics["accuracy"]),
        "precision": np.mean(metrics["precision"]),
        "recall": np.mean(metrics["recall"]),
        "f1": np.mean(metrics["f1"]),
    }

# Summary of results
print(f"\n{'=' * 50}")
print("SUMMARY OF RESULTS")
print(f"{'=' * 50}")
print(
    f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
)
print("-" * 70)
for name, metrics in sorted(
    results.items(), key=lambda x: x[1]["accuracy"], reverse=True
):
    print(
        f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}"
    )
