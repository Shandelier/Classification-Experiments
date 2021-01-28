#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

import enum
import numpy as np
import os  # to list files
import re  # to use regex
import csv  # to save some outputratio
import json
from numpy.lib.function_base import average, extract
from tqdm import tqdm
import utils as ut
from sklearn import neighbors, naive_bayes, svm, tree, neural_network
from sklearn import base
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA, KernelPCA

from imblearn.metrics import geometric_mean_score

random_state = 1410

# Initialize classifiers
classifiers = {
    'GNB': naive_bayes.GaussianNB(),
    'kNN': neighbors.KNeighborsClassifier(),
    'SVC-LIN': svm.SVC(gamma='scale', kernel='linear'),
    'SVC-RBF': svm.SVC(gamma='scale'),
    'CART': tree.DecisionTreeClassifier(),
}

# Choose metrics
used_metrics = {
    'ACC': metrics.accuracy_score,
    'BAC': metrics.balanced_accuracy_score,
    # 'APC': metrics.average_precision_score,
    # 'BSL': metrics.brier_score_loss,
    # 'CKS': metrics.cohen_kappa_score,
    'F1': metrics.f1_score,
    # 'HaL': metrics.hamming_loss,
    # 'HiL': metrics.hinge_loss,
    # 'JSS': metrics.jaccard_similarity_score,
    # 'LoL': metrics.log_loss,
    # 'MaC': metrics.matthews_corrcoef,
    # 'PS': metrics.precision_score,
    # 'RCS': metrics.recall_score,
    # 'AUC': metrics.roc_auc_score,
    # 'ZOL': metrics.zero_one_loss,
}


ds_paths = {
    'NO-EX': "datasets/",
    'PCA': "datasets_PCA/",
    'KPCA': "datasets_KPCA/",
    'Chi': "datasets_Chi/", }


res_paths = {
    'NO-EX': "results/",
    'PCA': "results_PCA/",
    'KPCA': "results_KPCA/",
    'Chi': "results_Chi/",
}


# Gather all the datafiles and filter them by tags
# tags_arr = ["binary", "multi-class", "multi-feature", "imbalanced"]
datasets = []
files = ut.dir2files("datasets/")
for file in files:
    #  TODO: intersecion of tags
    X, y, dbname, _ = ut.csv2Xy(file)
    datasets.append((X, y, dbname))


# Temporal tqdm disabler
disable = True
skf = model_selection.StratifiedKFold(n_splits=5)

# for hp in hyper_parameters:
#     for par_name in hp:
#         for fold in folds:
#             for metr in metrics:

for i, clf_par in enumerate(tqdm(hyper_parameters, desc="HP", ascii=True, position=0, leave=True)):
    # Prepare results cube
    rescube = np.zeros((len(datasets), len(
        hyper_parameters[clf_par]), len(used_metrics), 5))

    for c, par_name in enumerate(tqdm(hyper_parameters[clf_par], desc="PAR", ascii=True, position=1, leave=True, disable=disable)):
        X = datasets[0][0].copy()
        y = datasets[0][1].copy()

        for fold, (train, test) in enumerate(
            tqdm(skf.split(X, y), desc="FLD", ascii=True,
                 total=5, position=2, leave=True, disable=disable)
        ):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = base.clone(hyper_parameters[clf_par][par_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True, position=3, leave=True, disable=disable)):
                try:
                    score = used_metrics[metric_name](y_test, y_pred)
                    rescube[0, c, m, fold] = score
                except:
                    rescube[0, c, m, fold] = np.nan

    np.save("results_{}/rescube".format(clf_par), rescube)
    with open("results_{}/legend.json".format(clf_par), "w") as outfile:
        json.dump(
            {
                "datasets": [obj[2] for obj in datasets],
                "classifiers": list(hyper_parameters[clf_par]),
                "metrics": list(used_metrics.keys()),
                "folds": 5,
            },
            outfile,
            indent="\t",
        )
print("\n")
