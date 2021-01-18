import os
import argparse
import glob
import preprocess_utils as pre
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


parser = argparse.ArgumentParser()
parser.add_argument("--raw_dataset", type=str, default=r'./census.csv')
parser.add_argument("--datasets", type=str, default=r'./datasets')
parser.add_argument("--results", type=str, default=r'./results')
args = parser.parse_args()


def main():
    raw_dataset = args.raw_dataset
    datasets = args.datasets
    results = args.results

    # Create directories
    if not os.path.isdir(datasets):
        os.makedirs(datasets)
    if not os.path.isdir(results):
        os.makedirs(results)

    _NAMES = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race",
              "sex", "capital-gain", "capital-loss", "hours-per-week",
              "native-country", "income"]
    ds = pd.read_csv(raw_dataset, header=None, skiprows=1, names=_NAMES)

    ds = pre.clean_data(ds)
    ds = pre.standardize_data(ds)
    X, y = pre.split_data(ds)
    # One-hot encode the data and cast to Numpy array type
    X, y = pre.ohe_data(X, y)

    # prepare dictionary with extraction methods
    extr = {
        'NO_EXTR': "no extraction",
        'PCA': PCA(svd_solver='auto', random_state=420),
        'LDA': LinearDiscriminantAnalysis(),
        'KPCA': KernelPCA(n_components=X.shape[1], random_state=420, kernel='rbf', copy_X=False, eigen_solver='auto')}

    # iterate over extraction methods and save X in .csv with y labels
    for i, e in enumerate(extr):
        data = pre.extract(X, y, extr, e)
        np.savetxt(datasets+"/census_{}.csv".format(e), data, delimiter=",")

    files = sorted(list(glob.glob(datasets+"/*.csv")))
    for f in files:
        name = os.path.basename(f)
        print(name)


if __name__ == "__main__":
    main()
