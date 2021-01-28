#!/usr/bin/env python
# A script to analyze datasets available at w4k2/benchmark_datasets.

import os  # to list files
import csv  # to save some output
import numpy as np  # to calculate ratio
import utils as ut


if not os.path.isdir("datasets_"):
    print("NO DATASETS FOLDER")
    exit()
if not os.path.isdir("results_"):
    os.makedirs("results_")

# Gather all the datafiles
files = ut.dir2files("datasets_/")


# Iterate datafiles
with open("results_/datasets.csv", "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    # write header
    writer.writerow(["dataset", "samples", "features",
                     "classes", "ratio", "tags"])
    for file in files:
        # load dataset
        X, y, dbname, tags = ut.csv2Xy(file)

        # gather information
        tags = " ".join(tags)
        numberOfFeatures = X.shape[1]
        numberOfSamples = len(y)
        numberOfClasses = len(np.unique(y))

        # Calculate ratio
        ratio = [0.0] * numberOfClasses
        for y_ in y:
            ratio[y_] += 1
        ratio = [int(round(i / min(ratio))) for i in ratio]

        ratio = str(max(ratio))

        # write information
        writer.writerow(
            [dbname, numberOfSamples, numberOfFeatures, numberOfClasses, ratio, tags]
        )

        print(
            "%3i features, %5i samples, %2i classes, %3s ratio - %s (%s)"
            % (numberOfFeatures, numberOfSamples, numberOfClasses, ratio, dbname, tags)
        )
