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
from sklearn.decomposition import PCA

from imblearn.metrics import geometric_mean_score

# Initialize classifiers and extractions

classifiers_kernels = {
    'SVC-Lin': svm.SVC(kernel='linear'),
    'SVC-Sig': svm.SVC(kernel='sigmoid'),
    'SVC-RBF': svm.SVC(kernel='rbf'),
}

classifiers_Cs = {
    'SVC0.01': svm.SVC(C=.01),
    'SVC0.1': svm.SVC(C=.1),
    'SVC1': svm.SVC(C=1),
    'SVC10': svm.SVC(C=10),
    'SVC100': svm.SVC(C=100),
    'SVC1000': svm.SVC(C=1000),
}

classifiers_gammas = {
    'SVC1': svm.SVC(gamma=1),
    'SVC0.1': svm.SVC(gamma=.1),
    'SVC0.01': svm.SVC(gamma=.01),
    'SVC0.001': svm.SVC(gamma=.001),
    'SVC0.0001': svm.SVC(gamma=.0001),
}

hyper_parameters = {
    "Kernels": classifiers_kernels,
    "Cs": classifiers_Cs,
    "Gammas": classifiers_gammas,
}


# Choose metrics
used_metrics = {
    # "ACC": metrics.accuracy_score,
    "BAC": metrics.balanced_accuracy_score,
    # 'APC': metrics.average_precision_score,
    # 'BSL': metrics.brier_score_loss,
    # 'CKS': metrics.cohen_kappa_score,
    # "F1": f1_,
    # 'HaL': metrics.hamming_loss,
    # 'HiL': metrics.hinge_loss,
    # 'JSS': metrics.jaccard_similarity_score,
    # 'LoL': metrics.log_loss,
    # 'MaC': metrics.matthews_corrcoef,
    # 'PS': metrics.precision_score,
    # 'RCS': metrics.recall_score,
    # 'AUC': metrics.roc_auc_score,
    # 'ZOL': metrics.zero_one_loss,
    # 'GMEAN': geometric_mean_score
}

# create directories
for hp in hyper_parameters:
    if not os.path.isdir("results_"+hp):
        os.makedirs("results_"+hp)


# Gather all the datafiles and filter them by tags
datasets = []
X, y, dbname, _ = ut.csv2Xy(
    "/content/PWr-OB-Metrics/datasets/COVID19_PCA_8.csv")
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
