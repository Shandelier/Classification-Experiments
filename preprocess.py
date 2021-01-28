import os
import csv
import cv2
import time
import glob
import argparse
import operator
import numpy as np
import utils as ut

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA, KernelPCA


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str,
                    default='./curated-chest-xray-image-dataset-for-covid19')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--output_dataset_dir', type=str, default='./datasets')
args = parser.parse_args()

random_state = 1410

extractors = {
    'PCA': PCA(),
    'KPCA': KernelPCA(random_state=random_state),
    'Chi': SelectKBest(score_func=chi2),
}

# create directories
for ex in extractors:
    if not os.path.isdir("datasets_"):
        print("NO DATASETS FOLDER")
        exit()
    if not os.path.isdir("results_"):
        os.makedirs("results_")
    if not os.path.isdir("datasets_"+ex):
        os.makedirs("datasets_"+ex)
    if not os.path.isdir("results_"+ex):
        os.makedirs("results_"+ex)

# Gather all the datafiles and filter them by tags
datasets = []
files = ut.dir2files("datasets/")
for file in files:
    X, y, dbname, _ = ut.csv2Xy(file)
    datasets.append((X, y, dbname))

# extract
for i, ex in enumerate(extractors):
    for ds in datasets:
        X, y, dbname = ds
        half = round(X.shape[1]/2)
        if (ex == 'PCA'):
            X = PCA(n_components=half, svd_solver='auto').fit_transform(X)
        elif (ex == 'KPCA'):
            X = KernelPCA(n_components=half, random_state=random_state, kernel='rbf',
                          copy_X=False, eigen_solver='auto').fit_transform(X)
        elif (ex == 'Chi'):
            X = SelectKBest(score_func=chi2, k=half).fit_transform(X, y)
        y = y.reshape(y.shape[0], 1)
        data = np.concatenate((X, y), axis=1)
        np.savetxt("datasets_{}/{}.csv".format(ex, dbname),
                   X=data, delimiter=",")
