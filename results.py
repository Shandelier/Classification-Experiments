# This script read result cubes and perform statistic significance tests
from numpy.core.numeric import NaN
import utils as ut
import numpy as np
from scipy import stats
import latextabs as lt
from tabulate import tabulate

wil_stat = stats.wilcoxon
stud_stat = stats.ttest_ind
alfa = 0.05


res_paths = {
    'NO-EX': "results_/",
    'PCA': "results_PCA/",
    'KPCA': "results_KPCA/",
    'Chi': "results_Chi/",
}

for path in res_paths:
    # read columns and indices names
    legend = ut.json2object(res_paths[path]+"legend.json")
    datasets = legend["datasets"]
    classifiers = legend["classifiers"]
    metrics = legend["metrics"]
    folds = legend["folds"]
    rescube = np.load(res_paths[path]+"/rescube.npy")

    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))

    clf_headers = classifiers
    clf_indeces = np.array([classifiers])

    # ranks = np.zeros((len(metrics), len(datasets), len(classifiers)))

    for imet, met in enumerate(metrics):
        for ids, ds in enumerate(datasets):
            scores = rescube[ids, :, imet, :]

            for i in range(len(classifiers)):
                for j in range(len(classifiers)):
                    t_statistic[i, j], p_value[i, j] = stud_stat(
                        scores[i], scores[j])

            t_statistic_table = np.concatenate(
                (clf_indeces.T, t_statistic), axis=1)
            t_statistic_table = tabulate(
                t_statistic_table, clf_headers, floatfmt=".2f")

            advantage = np.zeros((len(classifiers), len(classifiers)))
            advantage[t_statistic > 0] = 1
            advantage_table = tabulate(np.concatenate(
                (clf_indeces.T, advantage), axis=1), clf_headers)

            significance = np.zeros(
                (len(classifiers), len(classifiers)))
            significance[p_value <= alfa] = 1
            significance_table = tabulate(np.concatenate(
                (clf_indeces.T, significance), axis=1), clf_headers)

            stat_better = significance * advantage
            stat_better_table = tabulate(np.concatenate(
                (clf_indeces.T, stat_better), axis=1), clf_headers)
            print(
                ">> [{}]:{}:{} << Statistically significantly better:\n".format(path, met, ds), stat_better_table)
