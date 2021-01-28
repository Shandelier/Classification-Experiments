#!/usr/bin/env python
from numpy.core.numeric import NaN
import utils as ut
import numpy as np
from scipy import stats
import latextabs as lt

# Parameters
used_test = stats.wilcoxon
used_p = 0.05

hyper_parameters = {
    "": None,
    "Cs": None,
    "Gammas": None,
}


for hp in hyper_parameters:
    results = "results_"+hp

    # Load results
    legend = ut.json2object(results+"/legend.json")
    datasets = legend["datasets"]
    classifiers = legend["classifiers"]
    metrics = legend["metrics"]
    folds = legend["folds"]
    rescube = np.load(results+"/rescube.npy")

    ranks = np.zeros((len(metrics), len(datasets), len(classifiers)))

    # First generate tables for each metric
    for mid, metric in enumerate(metrics):
        print("Metric %s [%i]" % (metric, mid))

        table_file = open(results+"/tab_%s.tex" % metric, "w")
        table_file.write(lt.header4classifiers(classifiers))

        for did, dataset in enumerate(datasets):
            print("| Dataset %10s [%i]" % (dataset, did))
            dataset = dataset.replace("_", "-")
            # print(dataset)
            # continue

            # Subtable is 2d (clf, fold)
            subtable = rescube[did, :, mid, :]

            # Check if metric was valid
            if np.isnan(subtable).any():
                print("Unvaild")
                continue

            # Scores as mean over folds
            scores = np.mean(subtable, axis=1)
            stds = np.std(subtable, axis=1)

            # ranks
            rank = stats.rankdata(scores, method='average')
            ranks[mid, did] = rank

            # Get leader and check dependency
            # dependency = np.zeros(len(classifiers)).astype(int)
            dependency = np.zeros((len(classifiers), len(classifiers)))

            for cida, clf_a in enumerate(classifiers):
                a = subtable[cida]
                for cid, clf in enumerate(classifiers):
                    b = subtable[cid]
                    test = used_test(a, b, zero_method="zsplit")
                    dependency[cida, cid] = test.pvalue > used_p

            print("dependency", dependency)
            print(scores)
            print("stds", stds)
            table_file.write(lt.row(dataset, scores, stds))
            table_file.write(lt.row_stats(dataset, dependency, scores, stds))
            # exit()

        table_file.write(lt.footer("Results for %s metric" % metric))
        table_file.close()

    for i, metric in enumerate(metrics):
        table_file = open(results+"/tab_%s_mean_ranut.tex" % metric, "w")
        table_file.write(lt.header4classifiers_ranks(classifiers))
        dependency2 = np.zeros((len(classifiers), len(classifiers)))
        for cida in range(len(classifiers)):
            a = ranks[i].T[cida]
            for cid in range(len(classifiers)):
                b = ranks[i].T[cid]
                test = used_test(a, b, zero_method="zsplit")
                dependency2[cida, cid] = test.pvalue > used_p

        print(dependency)
        print(np.mean(ranks[i], axis=0))
        table_file.write(lt.row_ranks(np.mean(ranks[i], axis=0)))
        table_file.write(lt.row_stats(dataset, dependency2,
                                      np.mean(ranks[i], axis=0), np.zeros((7))))
        table_file.write(
            lt.footer("Results for mean ranks according to %s metric" % metric))
        table_file.close()
