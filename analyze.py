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

# Initialize classifiers
classifiers = {
    'SVC': svm.SVC(gamma='scale'),
    'SVC+PCA': svm.SVC(gamma='scale'),
    'SVC+Chi': svm.SVC(gamma='scale'),
    # 'SVC': svm.SVC(gamma='scale'),
}

extractors = {
    'SVC': None,
    'SVC+PCA': PCA,
    'SVC+Chi': chi2
}

def extract(X, n_components, clf_name):
    if (clf_name == 'SVC'):
        return X
    elif (clf_name == 'SVC+PCA'):
        return PCA(n_components=n_components).fit_transform(X)
    elif (clf_name == 'SVC+Chi'):
        return SelectKBest(score_func=chi2, k=n_components).fit_transform(X,y)

# Choose metrics
used_metrics = {
    "ACC": metrics.accuracy_score,
    "BAC": metrics.balanced_accuracy_score,
    #'APC': metrics.average_precision_score,
    #'BSL': metrics.brier_score_loss,
    #'CKS': metrics.cohen_kappa_score,
    #'F1': metrics.f1_score,
    #'HaL': metrics.hamming_loss,
    #'HiL': metrics.hinge_loss,
    #'JSS': metrics.jaccard_similarity_score,
    #'LoL': metrics.log_loss,
    #'MaC': metrics.matthews_corrcoef,
    #'PS': metrics.precision_score,
    #'RCS': metrics.recall_score,
    #'AUC': metrics.roc_auc_score,
    #'ZOL': metrics.zero_one_loss,
}

# Gather all the datafiles and filter them by tags
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
# Prepare results cube
ds_feature_names = []
for c_n in components_arr:
    ds_feature_names.append("{}_{}_components".format(datasets[0][2], c_n))
print(
    "# Experiment on %i datasets, with %i estimators using %i metrics."
    % (len(ds_feature_names), len(classifiers), len(used_metrics))
)
rescube = np.zeros((len(ds_feature_names), len(classifiers), len(used_metrics), 5))


for i, n_components in enumerate(tqdm(components_arr, desc="DBS", ascii=True)):
    for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True)):
        X, y, dbname = datasets[0]
        skf = model_selection.StratifiedKFold(n_splits=5)
        X = extract(np.copy(X), n_components, clf_name)

        for fold, (train, test) in enumerate(
            tqdm(skf.split(X, y), desc="FLD", ascii=True, total=5)
        ):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = base.clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True)):
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

