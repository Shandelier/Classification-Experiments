# This script read result cubes and perform statistic significance_t tests
from numpy.core.numeric import NaN
import utils as ut
import numpy as np
from scipy import stats
from tabulate import tabulate

# wil_stat = stats.wilcoxon
wil_stat = stats.ranksums
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
    p_value_t = np.zeros((len(classifiers), len(classifiers)))

    wilcox_stat = np.zeros((len(classifiers), len(classifiers)))
    p_value_wil = np.zeros((len(classifiers), len(classifiers)))

    clf_headers = classifiers
    clf_indeces = np.array([classifiers])

    sig_better = np.full(
        shape=(1, len(classifiers)), fill_value="--------------")

    # ranks = np.zeros((len(metrics), len(datasets), len(classifiers)))

    for imet, met in enumerate(metrics):
        for ids, ds in enumerate(datasets):
            scores = rescube[ids, :, imet, :]

            for i in range(len(classifiers)):
                for j in range(len(classifiers)):
                    t_statistic[i, j], p_value_t[i, j] = stud_stat(
                        scores[i], scores[j])
                    wilcox_stat[i, j], p_value_wil[i, j] = wil_stat(scores[i], scores[j])

            t_statistic_table = np.concatenate(
                (clf_indeces.T, t_statistic), axis=1)
            # t_statistic_table = tabulate(
            #     t_statistic_table, clf_headers, floatfmt=".2f")
            w_statistic_table = np.concatenate(
                (clf_indeces.T, wilcox_stat), axis = 1)

            advantage_t = np.zeros((len(classifiers), len(classifiers)))
            advantage_t[t_statistic > 0] = 1
            # advantage_t_table = tabulate(np.concatenate(
            #     (clf_indeces.T, advantage_t), axis=1), clf_headers)
            advantage_w = np.zeros((len(classifiers), len(classifiers)))
            advantage_w[wilcox_stat > 0] = 1
            # advantage_w_table = tabulate(np.concatenate(
                # (clf_indeces.T, advantage_w), axis = 1), clf_headers)
            # print('advantage_w:\n', advantage_w)

            significance_t = np.zeros(
                (len(classifiers), len(classifiers)))
            significance_t[p_value_t <= alfa] = 1
            # significance_t_table = tabulate(np.concatenate(
            #     (clf_indeces.T, significance_t), axis=1), clf_headers)
            significance_w = np.zeros(
                (len(classifiers), len(classifiers)))
            significance_w[p_value_wil <= alfa] = 1
            # significance_w_table = tabulate(np.concatenate(
                # (clf_indeces.T, significance_w), axis = 1), clf_headers)
            # print(significance_w_table)

            stat_better_t = significance_t * advantage_t
            stat_better_w = significance_w * advantage_w
            # stat_better_t_table = tabulate(np.concatenate(
            #     (clf_indeces.T, stat_better_t), axis=1), clf_headers)
            # print(
            #     ">> [{}]:{}:{} << Statistically significantly better:\n".format(path, met, ds), stat_better_t_table)

            vec = np.full([len(classifiers)], fill_value="--------------")
            for i, name in enumerate(classifiers):
                for j, worse in enumerate(classifiers):
                    if (stat_better_t[i, j] == 1):
                        if (vec[i] == '--------------'):
                            vec[i] = worse
                        else:
                            vec[i] = vec[i] + "," + worse
            vec = vec.reshape(1, len(vec))
            sig_better = np.append(sig_better, vec, axis=0)

    sig_better[0, :] = classifiers
    indeces = np.concatenate(([path], datasets), axis=0)
    indeces = indeces.reshape(len(indeces), 1)
    sig_better = np.concatenate((indeces, sig_better), axis=1)
    print(tabulate(sig_better))
    print()
    print()