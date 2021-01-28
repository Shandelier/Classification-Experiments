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
    # 'SVC-LIN': svm.SVC(gamma='scale', kernel='linear'),
    # 'SVC-RBF': svm.SVC(gamma='scale'),
    'CART': tree.DecisionTreeClassifier(),
}

# Choose metrics
used_metrics = {
    'ACC': metrics.accuracy_score,
    # 'BAC': metrics.balanced_accuracy_score,
    # 'F1': metrics.f1_score,
}

ds_paths = {
    '': "datasets_/",
    'PCA': "datasets_PCA/",
    'KPCA': "datasets_KPCA/",
    'Chi': "datasets_Chi/", }

res_paths = {
    'NO-EX': "results_/",
    'PCA': "results_PCA/",
    'KPCA': "results_KPCA/",
    'Chi': "results_Chi/",
}

# create directories
for ex in ds_paths:
    if not os.path.isdir("datasets_"):
        print("NO DATASETS FOLDER")
        exit()
    if not os.path.isdir("results_"):
        os.makedirs("results_")
    if not os.path.isdir("datasets_"+ex):
        print("NO EXTRACTED DATASETS FOLDER")
        exit()
    if not os.path.isdir("results_"+ex):
        os.makedirs("results_"+ex)

for i, path in enumerate(ds_paths):

    # Gather all the datafiles and filter them by tags
    # tags_arr = ["binary", "multi-class", "multi-feature", "imbalanced"]
    datasets = []
    files = ut.dir2files("datasets_{}/".format(path))
    for file in files:
        #  TODO: intersecion of tags
        X, y, dbname, _ = ut.csv2Xy(file)
        datasets.append((X, y, dbname))

    # Temporal tqdm disabler
    disable = True
    skf = model_selection.StratifiedKFold(n_splits=5)

    # Prepare results cube
    rescube = np.zeros((len(datasets), len(classifiers), len(used_metrics), 5))

    for i, dataset in enumerate(tqdm(datasets, desc="DBS", ascii=True, position=0, leave=True)):
        # load dataset
        X, y, dbname = dataset

        # Folds
        skf = model_selection.StratifiedKFold(n_splits=5)
        for fold, (train, test) in enumerate(
            tqdm(skf.split(X, y), desc="FLD", ascii=True,
                 total=5, position=1, leave=True, disable=True)
        ):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True, position=2, leave=True, disable=True)):
                clf = base.clone(classifiers[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True, position=3, leave=True, disable=True)):
                    try:
                        score = used_metrics[metric_name](y_test, y_pred)
                        rescube[i, c, m, fold] = score
                    except:
                        rescube[i, c, m, fold] = np.nan
        # TODO: save paths
        np.save("results_{}/rescube".format(path), rescube)
        with open("results_{}/legend.json".format(path), "w") as outfile:
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
