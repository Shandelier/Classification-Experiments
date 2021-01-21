#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

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
classifiers = {
    'SVC':     svm.SVC(),
    'SVC+Chi': svm.SVC(),
    'SVC+PCA': svm.SVC(),
    # 'SVC': svm.SVC(gamma='scale'),
}


def extract(X, y, n_components, clf_name):
    if (clf_name == 'SVC'):
        return X
    elif (clf_name == 'SVC+PCA'):
        # print explained variance ratio for 5 best
        # if (n_components < 10):
        #     variance = PCA().fit(X).explained_variance_ratio_
        #     print("PCA best feature scores: ", variance[:5])
        return PCA(n_components=n_components).fit_transform(X)
    elif (clf_name == 'SVC+Chi'):
        selector = SelectKBest(score_func=chi2, k=n_components)
        best = selector.fit_transform(X, y)
        # print explained variance scores
        # if (n_components < 10):
        #     variance = selector.scores_
        #     print("Chi2 best feature scores: ", variance)
        return best


def f1_(y_true, y_pred):
    sth = metrics.f1_score(y_true, y_pred, average='macro')
    # TODO: delete
    # print(sth)
    return sth


# Choose metrics
used_metrics = {
    # "ACC": metrics.accuracy_score,
    "BAC": metrics.balanced_accuracy_score,
    # 'APC': metrics.average_precision_score,
    # 'BSL': metrics.brier_score_loss,
    # 'CKS': metrics.cohen_kappa_score,
    "F1": f1_,
    # 'HaL': metrics.hamming_loss,
    # 'HiL': metrics.hinge_loss,
    # 'JSS': metrics.jaccard_similarity_score,
    # 'LoL': metrics.log_loss,
    # 'MaC': metrics.matthews_corrcoef,
    # 'PS': metrics.precision_score,
    # 'RCS': metrics.recall_score,
    # 'AUC': metrics.roc_auc_score,
    # 'ZOL': metrics.zero_one_loss,
    'GMEAN': geometric_mean_score
}

# Gather all the datafiles and filter them by tags
# TODO: refactor it to select only one dataset
files = ut.dir2files("datasets/")
if (len(files) > 1):
    print("too many datasets, place only one in directory. Enter anything to continue on your own resposibility")
    input()
datasets = []
for file in files:
    X, y, dbname, _ = ut.csv2Xy(file)
    datasets.append((X, y, dbname))

#  Vector of selected features quantity in iteration
n_features = datasets[0][0].shape[1]
components_arr = np.arange(2, n_features+1, 2)
# prepare row names
ds_feature_names = []
for c_n in components_arr:
    ds_feature_names.append("{}_{}_components".format(datasets[0][2], c_n))

print(
    "# Experiment on %i datasets, with %i estimators using %i metrics."
    % (len(ds_feature_names), len(classifiers), len(used_metrics))
)

# Prepare results cube
rescube = np.zeros((len(ds_feature_names), len(
    classifiers), len(used_metrics), 5))
# Temporal tqdm disabler
disable = True
skf = model_selection.StratifiedKFold(n_splits=5)

for i, n_components in enumerate(tqdm(components_arr, desc="DBS", ascii=True, position=0, leave=True)):
    for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True, position=1, leave=True, disable=disable)):
        Xs, y, dbname = datasets[0]
        skf = model_selection.StratifiedKFold(n_splits=5)
        X = extract(Xs, y, n_components, clf_name)

        for fold, (train, test) in enumerate(
            tqdm(skf.split(X, y), desc="FLD", ascii=True,
                 total=5, position=2, leave=True, disable=disable)
        ):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = base.clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True, position=3, leave=True, disable=disable)):
                try:
                    score = used_metrics[metric_name](y_test, y_pred)
                    rescube[i, c, m, fold] = score
                except:
                    rescube[i, c, m, fold] = np.nan


np.save("results/rescube", rescube)
with open("results/legend.json", "w") as outfile:
    json.dump(
        {
            "datasets": [obj for obj in ds_feature_names],
            "classifiers": list(classifiers.keys()),
            "metrics": list(used_metrics.keys()),
            "folds": 5,
        },
        outfile,
        indent="\t",
    )

print("\n")
