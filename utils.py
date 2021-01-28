# A little module with some useful methods by Paweł Ksieniewicz.
import numpy as np
import os
import re
import json
import glob


def json2object(path):
    with open(path) as json_data:
        return json.load(json_data)


def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=",")
    ds = np.delete(ds, 0, 0)
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)
    dbname = path.split("/")[-1].split(".")[0]
    tags = tags4Xy(X, y)
    return X, y, dbname, tags


def dir2files(path, extension="csv"):
    return [
        path + x
        for x in os.listdir(path)
        if re.match("^([a-zA-Z0-9-_])+\.%s$" % extension, x)
    ]


def tags4Xy(X, y):
    tags = []
    numberOfFeatures = X.shape[1]
    numberOfSamples = len(y)
    numberOfClasses = len(np.unique(y))
    if numberOfClasses == 2:
        tags.append("binary")
    else:
        tags.append("multi-class")
    if numberOfFeatures >= 8:
        tags.append("multi-feature")

    # Calculate ratio
    ratio = [0.0] * numberOfClasses
    for y_ in y:
        ratio[y_] += 1
    ratio = [int(round(i / min(ratio))) for i in ratio]
    if max(ratio) > 4:
        tags.append("imbalanced")

    return tags


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def datasets_for_groups(ds_groups):
    datasets = []
    ds_dir = "keel"
    for group_idx, ds_group in enumerate(ds_groups):
        group_path = "%s/%s" % (ds_dir, ds_group)
        ds_list = sorted(os.listdir(group_path))
        for ds_idx, ds_name in enumerate(ds_list):
            if ds_name[0] == "." or ds_name[0] == "_":
                continue
            datasets.append((group_path, ds_name))
    return datasets


def load_dataset(dataset):
    group_path, ds_name = dataset
    # Load full dataset
    X, y = load_keel("%s/%s/%s.dat" % (group_path, ds_name, ds_name))
    X_, y_ = [], []
    # Load and process folds
    for i in range(1, 6):
        X_train, y_train = load_keel(
            "%s/%s/%s-5-fold/%s-5-%itra.dat"
            % (group_path, ds_name, ds_name, ds_name, i)
        )
        X_test, y_test = load_keel(
            "%s/%s/%s-5-fold/%s-5-%itst.dat"
            % (group_path, ds_name, ds_name, ds_name, i)
        )
        X_.append((X_train, X_test))
        y_.append((y_train, y_test))
    return (X, y, X_, y_)


def load_keel(string, separator=","):
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception

    s = filter(lambda e: e[0] != "@", s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:, :-1], dtype=float)
    d = {"positive": 1, "negative": 0}
    y = np.asarray(
        [d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip() for v in s]
    )

    return X, y


def merge_csvs(output, datasets):
    # Reading and combining data from partial CSV files
    featureFiles = sorted(list(glob.glob(output + "/*_features.csv")))
    labelFiles = sorted(list(glob.glob(output + "/*_labels.csv")))

    # gather data size for array init
    sizer = np.genfromtxt(featureFiles[0], delimiter=",")
    n_features = sizer.shape[1]
    X = np.empty([0, n_features])

    sizer = np.genfromtxt(labelFiles[0], delimiter=",")
    n_labels = sizer.shape
    y = np.empty([0, 1])

    for filename in featureFiles:
        data = np.genfromtxt(filename, delimiter=",")
        X = np.concatenate((X, data), axis=0)

    for filename in labelFiles:
        data = np.genfromtxt(filename, delimiter=",")
        y = np.append(y, data)

    a = X.shape
    b = y.shape

    ds = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    np.savetxt(datasets+"/COVID_19.csv", ds, delimiter=",")
    print("[INFO] CSVs merged")
