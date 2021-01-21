import os
import csv
import cv2
import time
import glob
import argparse
import operator
import numpy as np
import pandas as pd

from utils import merge_csvs, csv2Xy

from imutils import paths
from multiprocessing import Pool
from multiprocessing import cpu_count

# GLCM libs
from skimage.feature import greycoprops
from skimage.feature import greycomatrix

from datetime import datetime

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str,
                    default='./curated-chest-xray-image-dataset-for-covid19')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--output_dataset_dir', type=str, default='./datasets')
args = parser.parse_args()


# Function splitting data sets evenly between available processors
def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


def process_images(payload):

    # Features dictionary
    labels = np.asarray([['0'], ['1'], ['2'], ['3']])

    # Resize image target
    target_size = 1024

    # GLCM distances & angles (in radians)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, np.pi * 0.75]
    props = ["contrast", "dissimilarity",
             "homogeneity", "ASM", "energy", "correlation"]

    # features and labels
    X_partial = []
    y_partial = []

    # Loop over the image paths
    for imagePath in payload["input_paths"]:
        img = cv2.imread(imagePath)
        if img is not None:
            # Crop image to square (center-based)
            crop_dim = min(img.shape[0], img.shape[1])
            bounding = (crop_dim, crop_dim)
            start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
            end = tuple(map(operator.add, start, bounding))
            slices = tuple(map(slice, start, end))
            crop_img = img[slices]

            # INTER_CUBIC interpolation for enlarging images, INTER_AREA for shrinking
            interpolation = cv2.INTER_CUBIC if crop_img.shape[0] <= target_size else cv2.INTER_AREA
            dsize = (target_size, target_size)
            reshaped = cv2.resize(crop_img, dsize, interpolation)

            # Convert RGB to grayscale
            grayscale = cv2.cvtColor(reshaped, cv2.COLOR_BGR2GRAY)
            img_array = np.asarray(grayscale, dtype=np.uint8)

            # GLCM
            g_matrix = greycomatrix(
                img_array, distances, angles, normed=True, symmetric=True)
            img_features = np.ravel(
                [np.ravel(greycoprops(g_matrix, prop)) for prop in props]).T

            label = ""
            if "Normal" in imagePath:
                label = labels[0]
            elif "COVID-19" in imagePath:
                label = labels[1]
            elif "Pneumonia-Bacterial" in imagePath:
                label = labels[2]
            elif "Pneumonia-Viral" in imagePath:
                label = labels[3]

            X_partial.append(img_features)
            y_partial.append(label)

    # Dump partial feature extraction results to CSV file
    pd.DataFrame(X_partial).to_csv(
        payload["output_path"] + "_features.csv", header=None, index=None)
    pd.DataFrame(y_partial).to_csv(
        payload["output_path"] + "_labels.csv", header=None, index=None)


def main():
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    results_dir = args.results_dir
    output_dataset_dir = args.output_dataset_dir

    # Start time counter
    start = time.perf_counter()

    if not os.path.exists(dataset_dir):
        raise Exception(
            "[ERROR] Dataset directory {} not found.".format(dataset_dir))
    else:
        print("[INFO] Dataset path: {}".format(os.path.realpath(dataset_dir)))

    for directory in (results_dir, output_dir):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Makedir for results
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results_dir = os.path.join(results_dir, timestamp)
    os.makedirs(results_dir)

    #  Output dir cleanup
    files = glob.glob('output\*')
    for f in files:
        os.remove(f)

    # Get number of available CPU cores
    procs = cpu_count()
    procIDs = list(range(0, procs))

    print("[INFO] Preparing list of image paths")
    allImagePaths = sorted(list(paths.list_images(dataset_dir)))
    numImagesPerProc = len(allImagePaths) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))

    chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

    # Payload data for each thread
    payloads = []
    for (i, imagePaths) in enumerate(chunkedPaths):
        outputPath = os.path.sep.join([output_dir, "proc_{:02d}".format(i+1)])
        data = {
            "id": i,
            "input_paths": imagePaths,
            "output_path": outputPath
        }
        payloads.append(data)

    print("[INFO] Launching pool of {} processes".format(procs))
    print("[INFO] Image preprocessing ...")
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
    pool.close()
    pool.join()

    # Timer snapshot
    snapshot = time.perf_counter()
    print(
        f"[INFO] Image preprocessing finished in {snapshot - start:0.4f} seconds")

    merge_csvs(output_dir, output_dataset_dir)
    print(f"[INFO] CSVs files merged")


if __name__ == "__main__":
    main()
