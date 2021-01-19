#!/usr/bin/env python
import preprocess_utils as pre
import numpy as np
import os

from sklearn.datasets import make_classification


def gen_data(n_samples, n_features, n_classes, weights):
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_repeated=0,
        n_redundant=0,
        n_informative=3,
        flip_y=.05,
        random_state=420,
        n_clusters_per_class=2,
        weights=weights,
    )


# Create directories
datasets = r"./datasets"
results = r"./results"
if not os.path.isdir(datasets):
    os.makedirs(datasets)
if not os.path.isdir(results):
    os.makedirs(results)


n_samples = 2500
n_features = [100, 200, 300, 500, 1000, 2000, 5000]
n_classes = 2
print("2 balanced class")
for f in n_features:
    X, y = gen_data(n_samples, f, n_classes, [1])
    y = np.array([y]).T
    ds = np.concatenate((X, y), axis=1)
    np.savetxt(datasets+"/balanced_{}_atr.csv".format(f),
               ds, delimiter=",")

n_features = 50
n_classes = 2
ratio = {"1-9": [.1, .9],
         "2-8": [.2, .8],
         "3-7": [.3, .7],
         "4-6": [.4, .6],
         "5-5": [.5, .5]}
print("2 class")
for f in ratio:
    X, y = gen_data(n_samples, n_features, n_classes, ratio[f])
    y = np.array([y]).T
    ds = np.concatenate((X, y), axis=1)
    np.savetxt(datasets+"/2_class_{}_ratio.csv".format(f), ds,  delimiter=",")

n_classes = 3
n_features = 50
ratio = {"7-2-1": [.7, .2, .1],
         "6-3-1": [.6, .3, .1],
         "5-3-2": [.5, .3, .2],
         "4-4-2": [.4, .4, .2],
         "3-3-3": [.33, .33, .33]}
print("3 class")
for f in ratio:
    X, y = gen_data(n_samples, n_features, n_classes, ratio[f])
    y = np.array([y]).T
    ds = np.concatenate((X, y), axis=1)
    np.savetxt(datasets+"/3_class_{}_ratio.csv".format(f), ds, delimiter=",")
