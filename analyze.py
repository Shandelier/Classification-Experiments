#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

import numpy as np
import os  # to list files
import re  # to use regex
import csv  # to save some outputratio
import json
from numpy.lib.function_base import extract
from tqdm import tqdm
import utils as ut
from sklearn import neighbors, naive_bayes, svm, tree, neural_network
from sklearn import base
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA


# Initialize classifiers and extractions
classifiers = {
    "GNB": naive_bayes.GaussianNB(),
    "kNN": neighbors.KNeighborsClassifier(3),
    'SVC-LIN': svm.SVC(kernel="linear", C=0.025),
    'SVC': svm.SVC(kernel='rbf', gamma=2, C=1),
    'DTC': tree.DecisionTreeClassifier(max_depth=5),
    'MLP': neural_network.MLPClassifier(alpha=1, max_iter=1000)
}

# Choose metrics
used_metrics = {
    "ACC": metrics.accuracy_score,
    "BAC": metrics.balanced_accuracy_score,
    # 'APC': metrics.average_precision_score,
    # 'BSL': metrics.brier_score_loss,
    # 'CKS': metrics.cohen_kappa_score,
    # 'F1': metrics.f1_score,
    # 'HaL': metrics.hamming_loss,
    # 'HiL': metrics.hinge_loss,
    # 'JSS': metrics.jaccard_similarity_score,
    # 'LoL': metrics.log_loss,
    # 'MaC': metrics.matthews_corrcoef,
    # 'PS': metrics.precision_score,
    # 'RCS': metrics.recall_score,
    'AUC': metrics.roc_auc_score,
    # 'ZOL': metrics.zero_one_loss,
}

# Gather all the datafiles and filter them by tags
files = ut.dir2files("datasets/")
datasets = []
for file in files:
    X, y, dbname, tags = ut.csv2Xy(file)
    datasets.append((X, y, dbname))

# Prepare results cube
print(
    "# Experiment on %i datasets, with %i estimators using %i metrics."
    % (len(datasets), len(classifiers), len(used_metrics))
)
rescube = np.zeros((len(datasets), len(classifiers), len(used_metrics), 5))

# Iterate datasets
disable_tqdm = True
for i, dataset in enumerate(tqdm(datasets, desc="DBS", ascii=True, position=0, leave=True, disable=(not disable_tqdm))):
    # load dataset
    X, y, dbname = dataset

    # Folds
    skf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (train, test) in enumerate(
        tqdm(skf.split(X, y), desc="FLD", ascii=True, total=5,
             position=1, leave=True, disable=disable_tqdm)
    ):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True, position=2, leave=True, disable=disable_tqdm)):
            clf = base.clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True, position=3, leave=True, disable=disable_tqdm)):
                try:
                    score = used_metrics[metric_name](y_test, y_pred)
                    rescube[i, c, m, fold] = score
                except:
                    rescube[i, c, m, fold] = np.nan


np.save("results/rescube", rescube)
with open("results/legend.json", "w") as outfile:
    json.dump(
        {
            "datasets": [obj[2] for obj in datasets],
            "classifiers": list(classifiers.keys()),
            "metrics": list(used_metrics.keys()),
            "folds": 5,
        },
        outfile,
        indent="\t",
    )

print("\n")
