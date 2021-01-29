# This script read result cubes and perform statistic significance_t tests
from numpy.core.numeric import NaN
import utils as ut
import numpy as np
from scipy import stats
from tabulate import tabulate


def convert_to_latex_table(table, headers, first_column, use_float=False):
    try:
        print("\n\n================LATEX================\n")

        print(' & ', end='')
        print(' & '.join([header for header in headers]), end='')
        print('\\\\')

        k = 0
        for row in table:
            print(first_column[k] + ' & ', end='')
            if use_float:
                print(' & '.join(["{:.2f}".format(col) for col in row]), end='')
            else:
                print(' & '.join([str(int(col)) for col in row]), end='')
            print('\\\\\n\\hline')
            k += 1

        print("\n\n")
    except Exception:
        print("NO CAN DO, SORRY")
    

def wilcoxon_procedure(_rescube, _classifiers, _datasets, print_latex=False):
    mean_scores = np.mean(_rescube, axis=3)
    ranks = []
    for mean_score in mean_scores:
        ranks.append(stats.rankdata(mean_score).tolist())
    ranks = np.array(ranks)
    print('Ranks:\n', ranks)
    if print_latex:
        print("\n\n================LATEX================\n")
        convert_to_latex_table(ranks, _classifiers, _datasets, True)
        print("\n\n")

    mean_ranks = np.mean(ranks, axis=0)
    print()
    print('Mean ranks\n', _classifiers, '\n', mean_ranks)
    if print_latex:
        print("\n\n================LATEX================\n")
        convert_to_latex_table(mean_ranks, classifiers, ['mean rank'], True)
        print("\n\n")

    w_statistic = np.zeros((len(_classifiers), len(_classifiers)))
    p_value_wil = np.zeros((len(_classifiers), len(_classifiers)))
    for i in range(len(_classifiers)):
        for j in range(len(_classifiers)):
            w_statistic[i, j], p_value_wil[i, j] = wil_stat(ranks.T[i], ranks.T[j])

    names_column = np.expand_dims(np.array(_classifiers), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, _classifiers, floatfmt=".2f")
    p_value_wil_table = np.concatenate((names_column, p_value_wil), axis=1)
    p_value_wil_table = tabulate(p_value_wil_table, _classifiers, floatfmt=".2f")
    print('w-statistic:')
    print(w_statistic_table)
    if print_latex:
        convert_to_latex_table(w_statistic, _classifiers, _classifiers, True)

    print('p-value:')
    print(p_value_wil_table)
    if print_latex:
        convert_to_latex_table(p_value_wil, _classifiers, _classifiers, True)

    advantage = np.zeros((len(_classifiers), len(_classifiers)))
    advantage[w_statistic > 0] = 1
    advantage_w_table = tabulate(np.concatenate((names_column, advantage), axis=1), _classifiers)
    print('Advantage')
    print(advantage_w_table)
    if print_latex:
        convert_to_latex_table(advantage, _classifiers, _classifiers)

    significance = np.zeros((len(_classifiers), len(_classifiers)))
    significance[p_value_wil <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), _classifiers)
    print()
    print('Statistical significance ( alpha =', alfa, '):')
    print(significance_table)
    convert_to_latex_table(significance, _classifiers, _classifiers)


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

    wilcoxon_procedure(rescube, classifiers, datasets)

    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value_t = np.zeros((len(classifiers), len(classifiers)))

    clf_headers = classifiers
    clf_indeces = np.array([classifiers])

    sig_better = np.full(
        shape=(1, len(classifiers)), fill_value="------------------------------")

    for imet, met in enumerate(metrics):
        for ids, ds in enumerate(datasets):
            scores = rescube[ids, :, imet, :]

            for i in range(len(classifiers)):
                for j in range(len(classifiers)):
                    t_statistic[i, j], p_value_t[i, j] = stud_stat(
                        scores[i], scores[j])

            t_statistic_table = np.concatenate(
                (clf_indeces.T, t_statistic), axis=1)

            advantage_t = np.zeros((len(classifiers), len(classifiers)))
            advantage_t[t_statistic > 0] = 1

            significance_t = np.zeros(
                (len(classifiers), len(classifiers)))
            significance_t[p_value_t <= alfa] = 1

            stat_better_t = significance_t * advantage_t
            stat_better_t_table = tabulate(np.concatenate(
                (clf_indeces.T, stat_better_t), axis=1), clf_headers)
            print(
                ">> [{}]:{}:{} << Statistically significantly better:\n".format(path, met, ds), stat_better_t_table)
            print("\n\n================LATEX================\n")
            convert_to_latex_table(stat_better_t, classifiers, classifiers)
            print("\n\n")

            vec = np.full([len(classifiers)], fill_value="------------------------------")
            for i, name in enumerate(classifiers):
                for j, worse in enumerate(classifiers):
                    if stat_better_t[i, j] == 1:
                        if vec[i] == '------------------------------':
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