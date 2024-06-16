#!/usr/bin/env python3

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from helpers import get_dd
from math import log
from random import sample
from random import seed
from os import makedirs
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.svm import SVC

import functools as ft

import numpy as np
import pandas as pd
import re
import os
import json
import pickle
from nltools.stats import threshold
from nltools.plotting import plot_brain

# import plotly

import bids

# from nilearn import image as nimg
# from nilearn import plotting as nplot
# import seaborn as sns

# from bids import BIDSValidator
# import matplotlib
import matplotlib.pyplot as plt


from nltools.file_reader import onsets_to_dm

from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix

# from nltools.stats import find_spikes
# from nilearn.plotting import view_img, glass_brain, plot_stat_map

import nibabel as nib


# =============================================================
# == Contrasts
def number_contrast(glm_result, contrast, results_dir="../results/subj_contrasts"):
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    subj = glm_result["subj"]
    weight = contrast["weight"]
    # g1_dict = betas_ind(glm_result, contrast["g1"])
    # g2_dict = betas_ind(glm_result, contrast["g2"])

    betas = map(lambda x: betas_ind(glm_result, [x]), numbers)
    betas_averaged = Brain_Data(list(map(lambda x: sum(x.values()) / len(x), betas)))
    weight_normalized = np.array(weight) - np.mean(weight)
    # weight_normalized = [(x - 1) / len(weight) for x in weight]

    print(weight_normalized)
    # print(betas_averaged)

    out = betas_averaged * weight_normalized

    # out = sum(g1_dict.values()) / len(contrast["g1"]) - sum(g2_dict.values()) / len(
    #     contrast["g2"]
    # )
    # out.plot()
    # plt.savefig(
    #     os.path.join(results_dir, f"{subj}_" + contrast["name"] + "_contrast.png")
    # )
    # plt.close()
    return out


def view_number_contrast(
    l,
    group,
    run,
    contrast,
    results_dir="../results/group_contrasts",
):
    subjs = group["subjs"]
    group_desc = group["desc"]

    path_name = os.path.join(
        results_dir,
        f"{group_desc}_run-{run}_{contrast['name']}_contrasts.pkl",
    )
    if os.path.exists(path_name) and False:
        print("Loading contrasts for", contrast["name"])
        with open(path_name, "rb") as f:
            contrasts = pickle.load(f)
        return contrasts

    task = "matching"
    glm_curried = ft.partial(get_glm, l, task=task, run=run)
    contrast_type = ft.partial(number_contrast, contrast=contrast)

    # glm_results = list(map(glm_curried, subjs))
    # contrasts = list(map(contrast_type, glm_results))
    # to avoid OOM issues, use reduce
    contrasts = ft.reduce(
        lambda x, y: x + [contrast_type(y)], map(glm_curried, subjs), []
    )

    # contrasts = Brain_Data(contrasts)
    # res = contrasts.ttest(threshold_dict=th_param)
    # thr_t = res["thr_t"]
    # thr_t.plot(view="mni", colorbar=True)

    # with open(path_name, "wb") as f:
    #     pickle.dump(contrasts, f)

    return contrasts


def one_group_significance(
    l,
    contrast,
    group,
    threshold_dict={"unc": 0.001},
    results_dir="../results/group_contrasts",
):
    group_desc = group["desc"]
    thr_method = list(threshold_dict.keys())[0]
    thr_value = threshold_dict[thr_method]

    dir_name = os.path.join(
        results_dir, f"{group_desc}_{contrast['name']}_{thr_method}_{thr_value}_ttest"
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    path_name = os.path.join(
        dir_name,
        f"{group_desc}_{contrast['name']}_{thr_method}_{thr_value}_ttest.pkl",
    )
    if os.path.exists(path_name):
        print("Loading significance for", contrast["name"])
        with open(path_name, "rb") as f:
            significance = pickle.load(f)
        return significance

    contrasts = view_number_contrast(l, group, 1, contrast) + view_number_contrast(
        l, group, 2, contrast
    )
    contrasts = Brain_Data(contrasts)

    res = contrasts.ttest(threshold_dict=threshold_dict)
    thr_t = res["thr_t"]
    thr_t.plot(
        view="mni",
        colorbar=True,
        **{"vmin": -7.9, "vmax": 7.9},
    )
    plt.savefig(
        os.path.join(
            dir_name,
            f"{group_desc}_{contrast['name']}_{thr_method}_{thr_value}_ttest.png",
        )
    )
    plt.close()

    with open(path_name, "wb") as f:
        pickle.dump(thr_t, f)

    return thr_t


def two_sample_ttest(
    l,
    g1,
    g2,
    contrast,
    silent=False,
    design="contrast",
    p_value=0.001,
    results_dir="../results/two_ttest",
):
    # path_name = os.path.join(
    #     results_dir,
    #     f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{design}_data.pkl",
    # )
    # if os.path.exists(path_name) and not silent:
    #     with open(path_name, "rb") as f:
    #         data = pickle.load(f)

    if False:
        pass
    else:
        contrasts1 = view_number_contrast(l, g1, 1, contrast) + view_number_contrast(
            l, g1, 2, contrast
        )
        contrasts2 = view_number_contrast(l, g2, 1, contrast) + view_number_contrast(
            l, g2, 2, contrast
        )
        data = Brain_Data(contrasts1 + contrasts2)
        n1 = len(contrasts1)
        n2 = len(contrasts2)
        dm = np.zeros((n1 + n2, 2))
        if design == "contrast":
            dm[:, 0] = 1
            dm[:n1, 1] = -1
            dm[n1:, 1] = 1
        elif design == "group":
            dm[:n1, 0] = 1  # Group 1
            dm[n1:, 1] = 1  # Group 2

        dm = pd.DataFrame(dm, columns=[g1["desc"], g2["desc"]])
        data.X = dm
        labels = [g1["desc"]] * n1 + [g2["desc"]] * n2
        data.Y = pd.DataFrame(labels)
        if silent:
            return data

    # Fit the regression model
    stats = data.regress()

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Threshold the t-values by p-value
    # plot the t-values thresholded by p-value
    if design == "contrast":
        t_map = stats["t"][1]  # t-map for the group difference
        p_map = stats["p"][1]  # p-map for the group difference
        thr_method = "unc"
        thr_value = p_value
        thresholded_t_map = threshold(t_map, p_map, p_value)
        if not silent:
            thresholded_t_map.plot(view="mni", colorbar=True)
            plt.savefig(
                os.path.join(
                    results_dir,
                    f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{thr_method}_{thr_value}_{design}_ttest.png",
                )
            )
            plt.close()
    elif design == "group":
        t_map = stats["t"][1]
        p_map = stats["p"][1]
        thr_method = "unc"
        thr_value = p_value
        thresholded_t_map = threshold(t_map, p_map, p_value)

        if not silent:
            thresholded_t_map.plot(view="mni", colorbar=True)
            plt.savefig(
                os.path.join(
                    results_dir,
                    f"{g2['desc']}_{contrast['name']}_{thr_method}_{thr_value}_{design}_ttest.png",
                )
            )
            plt.close()
        t_map0 = stats["t"][0]
        p_map0 = stats["p"][0]
        thresholded_t_map0 = threshold(t_map0, p_map0, p_value)
        thresholded_t_map0.plot(view="mni", colorbar=True)
        plt.savefig(
            os.path.join(
                results_dir,
                f"{g1['desc']}_{contrast['name']}_{thr_method}_{thr_value}_{design}_ttest.png",
            )
        )
        plt.close()

    # if not silent:
    #     with open(path_name, "wb") as f:
    #         pickle.dump(data, f)

    return data


def classify_contrast(
    l,
    g1,
    g2,
    contrast,
    design="contrast",
    algorithm="svm",
    algo_args={"kernel": "linear"},
    n_folds=3,
    p_value=0.001,
    results_dir="../results/classifier",
):
    path_name = os.path.join(
        results_dir,
        f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_classifier.pkl",
    )
    if os.path.exists(path_name):
        with open(path_name, "rb") as f:
            data = pickle.load(f)
    else:
        pass
    data = two_sample_ttest(l, g1, g2, contrast)

    stats = data.regress()

    # Threshold the t-values by p-value
    t_map = stats["t"][1]  # t-map for the group difference
    p_map = stats["p"][1]  # p-map for the group difference
    thresholded_t_map = threshold(t_map, p_map, p_value)

    # switch to nifti
    mask = thresholded_t_map.to_nifti()
    # turn into true/false
    bool_mask = mask.get_fdata() != 0
    mask = nib.Nifti1Image(bool_mask.astype(np.float32), affine=mask.affine)

    # visualize the mask
    thresholded_t_map.plot(view="mni", colorbar=True)
    plt.savefig(
        os.path.join(
            results_dir,
            f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_p-{p_value}_n-{n_folds}_mask.png",
        )
    )
    plt.close()
    data = data.apply_mask(mask)
    print("Length of data is " + str(len(data.Y)))
    stats = data.predict(
        algorithm=algorithm,
        cv_dict={"type": "kfolds", "n_folds": n_folds, "n": len(data.Y)},
        plot=False,
        **algo_args,
    )
    with open(path_name, "wb") as f:
        pickle.dump(data, f)

    return data


def grouped_classify(
    l,
    g1,
    g2,
    contrast,
    sd,
    algorithm="svm",
    mask_sample_size=5,
    algo_args={"kernel": "linear"},
    n_folds=7,
    p_value=0.001,
    results_dir="../results/grouped_classifier",
):
    seed(sd)
    new_path_name = os.path.join(
        results_dir,
        f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_permutation.pkl",
    )
    # path_name = os.path.join(
    #     results_dir,
    #     f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_classifier.pkl",
    # )
    data_name = os.path.join(
        results_dir,
        f"{g1['desc']}_{g2['desc']}_{contrast['name']}_data.pkl",
    )
    if os.path.exists(new_path_name):
        with open(new_path_name, "rb") as f:
            result = pickle.load(f)
            return result

    elif os.path.exists(data_name):
        with open(data_name, "rb") as f:
            data = pickle.load(f)
            mask_g1 = sample(g1["subjs"], mask_sample_size)
            mask_g2 = sample(g2["subjs"], mask_sample_size)

    else:
        mask_g1 = sample(g1["subjs"], mask_sample_size)
        mask_g2 = sample(g2["subjs"], mask_sample_size)
        print(mask_g1)
        svm_g1 = [x for x in g1["subjs"] if x not in mask_g1]
        svm_g2 = [x for x in g2["subjs"] if x not in mask_g2]

        mask_g1 = {"desc": "mask_g1", "subjs": mask_g1}
        mask_g2 = {"desc": "mask_g2", "subjs": mask_g2}
        svm_g1 = {"desc": g1["desc"] + "_svm", "subjs": svm_g1}
        svm_g2 = {"desc": g2["desc"] + "_svm", "subjs": svm_g2}

        data = two_sample_ttest(l, svm_g1, svm_g2, contrast, silent=True)
        mask_dat = two_sample_ttest(l, mask_g1, mask_g2, contrast, silent=False)

        stats = mask_dat.regress()

        # Threshold the t-values by p-value
        t_map = stats["t"][1]  # t-map for the group difference
        p_map = stats["p"][1]  # p-map for the group difference
        thresholded_t_map = threshold(t_map, p_map, p_value)

        # switch to nifti
        mask = thresholded_t_map.to_nifti()
        # turn into true/false
        bool_mask = mask.get_fdata() != 0
        mask = nib.Nifti1Image(bool_mask.astype(np.float32), affine=mask.affine)

        # visualize the mask
        thresholded_t_map.plot(view="mni", colorbar=True)
        plt.savefig(
            os.path.join(
                results_dir,
                f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_p-{p_value}_n-{n_folds}_mask.png",
            )
        )
        plt.close()
        data = data.apply_mask(mask)
        with open(data_name, "wb") as f:
            pickle.dump(data, f)
    print("Length of data is " + str(len(data.Y)))

    clf = SVC(kernel="linear", random_state=7)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=0)

    score, perm_scores, pvalue = permutation_test_score(
        clf, data.data, data.Y, scoring="accuracy", cv=cv, n_permutations=1000
    )

    fig, ax = plt.subplots()

    ax.hist(perm_scores, bins=20, density=True)
    ax.axvline(score, ls="--", color="r")
    score_label = (
        f"Cross Validation Score \n data: {score:.2f}\n(p-value: {pvalue:.3f})"
    )
    ax.text(0.3, 4.5, score_label, fontsize=9)
    ax.set_xlabel("Accuracy score")
    _ = ax.set_ylabel("Probability density")

    plt.savefig(
        os.path.join(
            results_dir,
            f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_p-{p_value}_n-{n_folds}_permutation.png",
        )
    )
    plt.close()

    stats = {
        "score": score,
        "perm_scores": perm_scores,
        "pvalue": pvalue,
    }

    # stats = data.predict(
    #     algorithm=algorithm,
    #     cv_dict={"type": "kfolds", "n_folds": n_folds, "n": len(data.Y)},
    #     plot=False,
    #     **algo_args,
    # )

    result = {"mask_group": [mask_g1, mask_g2], "stats": stats}
    # with open(path_name, "wb") as f:
    #     pickle.dump(result, f)

    with open(new_path_name, "wb") as f:
        pickle.dump(result, f)

    return result


def run_grouped_classify(
    l,
    g1,
    g2,
    contrast,
    mask_sample_size=5,
    samples=10,
    rd_seed=42,
    n_folds=7,
    p_value=0.001,
    results_dir="../results/grouped_classifier/run",
):
    results_dir += str(rd_seed)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # seed(rd_seed)
    path_name = os.path.join(
        results_dir,
        "f{g1['desc']}_{g2['desc']}_{contrast['name']}_n-{n_folds}_p-{p_value}_classifier.pkl",
    )
    if os.path.exists(path_name) and False:
        with open(path_name, "rb") as f:
            results = pickle.load(f)
        return results
    results = []
    for i in range(samples):
        inner_path_name = os.path.join(results_dir, f"sample_{i}")
        if not os.path.exists(inner_path_name):
            os.makedirs(inner_path_name)

        sd = rd_seed * 100 + i
        res = grouped_classify(
            l,
            g1,
            g2,
            contrast,
            sd=sd,
            mask_sample_size=mask_sample_size,
            n_folds=n_folds,
            p_value=p_value,
            results_dir=inner_path_name,
        )
        results.append(res)
    with open(path_name, "wb") as f:
        pickle.dump(results, f)
    return results


def searchlight(
    l,
    g1,
    g2,
    contrast,
    algorithm="svm",
    algo_args={"kernel": "linear"},
    n_folds=3,
    p_value=0.001,
    results_dir="../results/searchlight",
):
    path_name = os.path.join(
        results_dir,
        f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_classifier.pkl",
    )
    if os.path.exists(path_name):
        with open(path_name, "rb") as f:
            data = pickle.load(f)
        return data

    data = two_sample_ttest(l, g1, g2, contrast)

    stats = data.regress()

    # Threshold the t-values by p-value
    t_map = stats["t"][1]  # t-map for the group difference
    p_map = stats["p"][1]  # p-map for the group difference
    thresholded_t_map = threshold(t_map, p_map, p_value)

    # switch to nifti
    mask = thresholded_t_map.to_nifti()
    # turn into true/false
    bool_mask = mask.get_fdata() != 0
    mask = nib.Nifti1Image(bool_mask.astype(np.float32), affine=mask.affine)

    # visualize the mask
    thresholded_t_map.plot(view="mni", colorbar=True)
    plt.savefig(
        os.path.join(
            results_dir,
            f"{g1['desc']}_{g2['desc']}_{contrast['name']}_{algo_args['kernel']}_mask.png",
        )
    )
    plt.close()

    # data = data.apply_mask(mask)
    stats = data.predict_multi(
        algorithm=algorithm,
        cv_dict={"type": "kfolds", "n_folds": n_folds, "n": len(data.Y)},
        method="searchlight",
        process_mask=mask,
        **algo_args,
    )
    with open(path_name, "wb") as f:
        pickle.dump(data, f)

    return data


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"
    run = 1
    proc_space = "MNI152NLin2009cAsym"

    subjs = l.get_subjects()
    error_subjs = ["0192", "0487", "1085"]

    subjs = [subj for subj in subjs if subj not in error_subjs]

    toy_group = {"desc": "toy", "subjs": subjs[:5]}
    toy10_group = {"desc": "toy-10", "subjs": subjs[:10]}
    full_group = {"desc": "full", "subjs": subjs}

    dd_ta = get_dd(full_group)
    dd_group = {"desc": "DD_full", "subjs": dd_ta["DD"]}
    ta_group = {"desc": "TA_full", "subjs": dd_ta["TA"]}

    groups = [full_group, dd_group, ta_group]

    # even_odd = {
    #     "name": "odd_even",
    #     "g1": ["1", "3", "5", "7", "9"],
    #     "g2": ["2", "4", "6", "8"],
    # }
    even_odd = {
        "name": "even_odd",
        "weight": [1, -1, 1, -1, 1, -1, 1, -1, 1],
    }
    # big_small = {
    #     "name": "big_small",
    #     "g1": ["5", "6", "7", "8", "9"],
    #     "g2": ["1", "2", "3", "4"],
    # }

    linear_relation = {
        "name": "linear_relation",
        "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    new_linear_relation = {
        "name": "new_linear_relation",
        "weight": list(map(lambda x: 1 / x, [1, 2, 3, 4, 5, 6, 7, 8, 9])),
    }

    rev_linear_relation = {
        "name": "reverse_linear_relation",
        "weight": [9, 8, 7, 6, 5, 4, 3, 2, 1],
    }

    log_relation = {
        "name": "log_relation",
        "weight": list(map(log, [1, 2, 3, 4, 5, 6, 7, 8, 9])),
    }

    new_log_relation = {
        "name": "new_log_relation",
        "weight": [0.01] + list(map(lambda x: 1 / log(x), [2, 3, 4, 5, 6, 7, 8, 9])),
    }

    # even_odd_contrast1 = ft.partial(view_number_contrast, l, run=1, contrast=even_odd)
    # even_odd_contrast2 = ft.partial(view_number_contrast, l, run=2, contrast=even_odd)
    # big_small_contrast1 = ft.partial(view_number_contrast, l, run=1, contrast=big_small)
    # big_small_contrast2 = ft.partial(view_number_contrast, l, run=2, contrast=big_small)

    # even_odd1 = list(map(even_odd_contrast1, groups))
    # even_odd2 = list(map(even_odd_contrast2, groups))
    # big_small1 = list(map(big_small_contrast1, groups))
    # big_small2 = list(map(big_small_contrast2, groups))

    # two_sample_ttest(
    #     l, dd_group, ta_group, linear_relation, design="contrast", p_value=0.005
    # )
    # two_sample_ttest(
    #     l, dd_group, ta_group, linear_relation, design="contrast", p_value=0.01
    # )
    # two_sample_ttest(
    #     l, dd_group, ta_group, linear_relation, design="contrast", p_value=0.05
    # )

    num_1 = {
        "name": "num_1",
        "weight": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    num_2 = {
        "name": "num_2",
        "weight": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    }
    num_3 = {
        "name": "num_3",
        "weight": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    }
    num_4 = {
        "name": "num_4",
        "weight": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    }
    num_5 = {
        "name": "num_5",
        "weight": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    }
    num_6 = {
        "name": "num_6",
        "weight": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    }
    num_7 = {
        "name": "num_7",
        "weight": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    }
    num_8 = {
        "name": "num_8",
        "weight": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    }
    num_9 = {
        "name": "num_9",
        "weight": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    # num_contrasts = [num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9]
    # num_tsttest = ft.partial(
    #     two_sample_ttest,
    #     l,
    #     dd_group,
    #     ta_group,
    #     design="contrast",
    #     p_value=0.001,
    #     results_dir="../results/two_ttest/num_ttest",
    # )

    # list(map(num_tsttest, num_contrasts))

    # my_ttest = ft.partial(
    #     one_group_significance,
    #     l,
    #     threshold_dict={"unc": 0.001},
    #     results_dir="../results/group_contrasts/new",
    # )
    # my_ttest(new_log_relation, full_group)
    # my_ttest(new_log_relation, dd_group)
    # my_ttest(new_log_relation, ta_group)
    # my_ttest(linear_relation, full_group)
    # my_ttest(linear_relation, dd_group)
    # my_ttest(linear_relation, ta_group)
    # my_ttest(even_odd, full_group)
    # my_ttest(even_odd, dd_group)
    # my_ttest(even_odd, ta_group)
    # my_ttest(big_small, full_group)
    # my_ttest(big_small, dd_group)
    # my_ttest(big_small, ta_group)

    # two_sample_ttest(l, dd_group, ta_group, new_log_relation)
    # two_sample_ttest(l, dd_group, ta_group, new_linear_relation)

    # linear_contrast1 = ft.partial(
    #     view_number_contrast, l, run=1, contrast=linear_relation
    # )
    # linear_contrast2 = ft.partial(
    #     view_number_contrast, l, run=2, contrast=linear_relation
    # )

    # linear1 = list(map(linear_contrast1, groups))
    # linear2 = list(map(linear_contrast2, groups))

    # linear_classifier = ft.partial(
    #     classify_contrast,
    #     l,
    #     algorithm="svm",
    #     p_value=0.001,
    #     n_folds=7,
    #     algo_args={"kernel": "linear"},
    # )

    # data = linear_classifier(dd_group, ta_group, linear_relation)

    # stats = data.predict(
    #     algorithm="svm",
    #     cv_dict={"type": "kfolds", "n_folds": 7, "n": len(data.Y)},
    #     plot=False,
    #     **{"kernel": "linear"},
    # )
    # print(stats)

    result1 = run_grouped_classify(
        l,
        dd_group,
        ta_group,
        linear_relation,
        mask_sample_size=20,
        samples=10,
        rd_seed=56,
        n_folds=7,
        p_value=0.01,
    )
    pvals = list(map(lambda x: x["stats"]["pvalue"], result1))
    print(pvals)
    # [0.11488511488511488, 0.4905094905094905, 0.36163836163836166, 0.1918081918081918, 0.2017982017982018, 0.1108891108891109, 0.08091908091908091, 0.23476523476523475, 0.4095904095904096, 0.43656343656343655]

    scores = list(map(lambda x: x["stats"]["score"], result1))
    print(scores)
    # [0.6428571428571429, 0.5612244897959183, 0.5765306122448981, 0.6198979591836735, 0.6173469387755102, 0.6403061224489797, 0.6607142857142857, 0.6173469387755102, 0.5637755102040816, 0.5561224489795917]

    # result2 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=15,
    #     samples=10,
    #     rd_seed=43,
    #     n_folds=7,
    #     p_value=0.005,
    # )
    # result3 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=15,
    #     samples=10,
    #     rd_seed=42,
    #     n_folds=7,
    #     p_value=0.01,
    # )

    # result4 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=15,
    #     samples=10,
    #     rd_seed=39,
    #     n_folds=7,
    #     p_value=0.05,
    # )

    # result5 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=15,
    #     samples=10,
    #     rd_seed=13,
    #     n_folds=5,
    #     p_value=0.01,
    # )

    # result5 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=12,
    #     samples=10,
    #     rd_seed=95,
    #     n_folds=7,
    #     p_value=0.005,
    # )

    # result6 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=12,
    #     samples=10,
    #     rd_seed=15,
    #     n_folds=7,
    #     p_value=0.01,
    # )

    # results = [result1, result2, result3, result4, result5, result6]

    # mcr_xvals = list(
    #     map(lambda x: list(map(lambda y: y["stats"]["mcr_xval"], x)), results)
    # )
    # avg_mcr = list(map(lambda x: sum(x) / len(x), mcr_xvals))
    # print(avg_mcr)

    # [[0.58, 0.54, 0.5, 0.58, 0.5, 0.56, 0.48, 0.56, 0.48, 0.48],
    # [0.5428571428571428, 0.5571428571428572, 0.5, 0.42857142857142855, 0.4857142857142857, 0.44285714285714284, 0.4857142857142857, 0.4857142857142857, 0.45714285714285713, 0.5714285714285714],
    # [0.5285714285714286, 0.45714285714285713, 0.4714285714285714, 0.4142857142857143, 0.5, 0.4142857142857143, 0.4142857142857143, 0.6, 0.38571428571428573, 0.44285714285714284],
    # [0.4142857142857143, 0.38571428571428573, 0.5142857142857142, 0.5142857142857142, 0.4714285714285714, 0.38571428571428573, 0.5, 0.35714285714285715, 0.44285714285714284, 0.5428571428571428],
    # [0.43902439024390244, 0.5609756097560976, 0.45121951219512196, 0.36585365853658536, 0.45121951219512196, 0.5365853658536586, 0.5121951219512195, 0.5, 0.4268292682926829, 0.4146341463414634],
    # [0.4146341463414634, 0.524390243902439, 0.45121951219512196, 0.3780487804878049, 0.4268292682926829, 0.524390243902439, 0.5121951219512195, 0.36585365853658536, 0.4878048780487805, 0.4268292682926829]]

    # avg_mcr_xvals = list(map(lambda x: sum(x) / len(x), mcr_xvals))
    # print(avg_mcr_xvals)

    # [0.5260000000000001, 0.4957142857142857, 0.4628571428571429, 0.45285714285714285, 0.46585365853658545, 0.4512195121951219]
    # result7 = run_grouped_classify(
    #     l,
    #     dd_group,
    #     ta_group,
    #     linear_relation,
    #     mask_sample_size=12,
    #     samples=10,
    #     rd_seed=10,
    #     n_folds=7,
    #     p_value=0.05,
    # )

# result8 = run_grouped_classify(
#     l,
#     dd_group,
#     ta_group,
#     linear_relation,
#     mask_sample_size=12,
#     samples=10,
#     rd_seed=14,
#     n_folds=5,
#     p_value=0.01,
# )

# linear_classifier(dd_group, ta_group, new_linear_relation)
# 0.36 whole brain n_folds 3
# 0.69 at mask 0.001 n_folds 3
# 0.56 at mask 0.001 n_folds 5 log
# 0.58 at mask 0.001 n_folds 7 log
# 0.72 at mask 0.01 n_folds 3
# 0.68 at mask 0.05 n_folds 3
# 0.74 at mask 0.01 n_folds 4
# 0.82 at mask 0.01 n_folds 5
# 0.84 at mask 0.05 n_folds 5
# 0.74 at mask 0.05 n_folds 7 log
# 0.82 at mask 0.1 n_folds 5
# 0.83 at mask 0.01 n_folds 7
# 0.79 at mask 0.005 n_folds 7
# 0.67 at mask 0.005 n_folds 7 log
# 0.82 at mask 0.01 n_folds 10
# 0.72 at mask 0.01 n_folds 10

# my_ttest(rev_linear_relation, full_group)
# my_ttest(rev_linear_relation, dd_group)
# my_ttest(rev_linear_relation, ta_group)
# two_sample_ttest(l, dd_group, ta_group, rev_linear_relation)

# linear_searchlight = ft.partial(
#     searchlight, l, algorithm="svm", algo_args={"kernel": "linear"}
# )

# linear_searchlight(dd_group, ta_group, linear_relation)
