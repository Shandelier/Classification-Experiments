# Little module for census data preprocessing by Karol Kłusek
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def clean_data(ds):
    # Replace all " ?" with NaN and then drop rows where NaN appears
    ds_clean = ds[ds.workclass != '?']
    print("Number of instances removed:", len(ds)-len(ds_clean))
    return ds_clean


def standardize_data(ds):
    numerical_col = ["age", "fnlwgt", "education-num", "capital-gain",
                     "capital-loss", "hours-per-week"]
    scaler = StandardScaler()
    # TODO: 3 sigma scaler
    ds[numerical_col] = scaler.fit_transform(ds[numerical_col])
    return ds


def split_data(ds):
    y = ds["income"]
    X = ds.drop("income", axis=1)
    return X, y


def ohe_data(X, y):
    data_ohe = pd.get_dummies(X)
    X = data_ohe.iloc[:len(X)]
    y = y.replace(['<=50K', '>50K'], [0, 1])

    # Cast to Numpy
    nX = np.squeeze(np.array([X]))
    ny = np.array([y]).T
    return nX, ny


def extract(X, y, extr, type):
    print(type)
    if (type == 'PCA'):
        X = extr[type].fit_transform(X)
    elif (type == 'LDA'):
        X = extr[type].fit_transform(X, np.squeeze(y))
    elif (type == 'KPCA'):
        # KPCA requires too much RAM to run the whole set
        X, y = resample(X, y, n_samples=int(len(y)*.1), random_state=420)
        X = extr[type].fit_transform(X, np.squeeze(y))

    # concatenating with y set
    return np.concatenate((X, y), axis=1)
