#!/usr/bin/env python3

from subject_glm import glm_ind
from subject_glm import betas_ind
from subject_glm import shape_ind
from subject_glm import estimate_subj_glm as get_glm
from helpers import groups
import bids
import os
import pickle
import functools as ft
from nltools.data import Brain_Data
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nltools.plotting import scatterplot, plot_interactive_brain, plot_brain


def number_baseline(glm_result, number, results_dir="../results/number_baseline"):
    number_betas = list(betas_ind(glm_result, [number]).values())

    print("Number betas", number_betas)
    shape_beta = list(shape_ind(glm_result).values())[0]
    print("Shape beta", shape_beta)
    contrast = sum(number_betas) / len(number_betas) - shape_beta
    return contrast


def group_number_baseline(
    l, group, run, number, results_dir="../results/number_baseline"
):
    subjs = group["subjs"]
    group_desc = group["desc"]

    path_name = os.path.join(
        results_dir,
        f"{group_desc}_run-{run}_{number}_contrasts.pkl",
    )
    if os.path.exists(path_name):
        print("Loading contrasts for", number)
        with open(path_name, "rb") as f:
            contrasts = pickle.load(f)
        return contrasts

    task = "matching"
    glm_curried = ft.partial(get_glm, l, task=task, run=run)
    contrast_type = ft.partial(number_baseline, number=number)

    # glm_results = list(map(glm_curried, subjs))
    # contrasts = list(map(contrast_type, glm_results))
    # to avoid OOM issues, use reduce
    contrasts = ft.reduce(
        lambda x, y: x + [contrast_type(y)], map(glm_curried, subjs), []
    )

    with open(path_name, "wb") as f:
        pickle.dump(contrasts, f)

    return contrasts


def number_activation(
    l,
    number,
    group,
    threshold_dict={"unc": 0.001},
    results_dir="../results/number_baseline",
):
    group_desc = group["desc"]
    thr_method, thr_value = list(threshold_dict.items())[0]
    dir_name = os.path.join(
        results_dir, f"{group_desc}_{number}_{thr_method}_{thr_value}_ttest"
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    path_name = os.path.join(
        dir_name,
        f"{group_desc}_{number}_{thr_method}_{thr_value}_ttest.pkl",
    )
    if os.path.exists(path_name):
        print("Loading significance for", number)
        with open(path_name, "rb") as f:
            thr_t = pickle.load(f)
    else:
        contrasts = group_number_baseline(l, group, 1, number) + group_number_baseline(
            l, group, 2, number
        )

        contrasts = Brain_Data(contrasts)

        res = contrasts.ttest(threshold_dict=threshold_dict)
        thr_t = res["thr_t"]
        with open(path_name, "wb") as f:
            pickle.dump(thr_t, f)

    thr_t.plot(view="mni", colorbar=True)
    plt.savefig(
        os.path.join(
            dir_name,
            f"{group_desc}_{number}_{thr_method}_{thr_value}_ttest.png",
        )
    )
    plt.close()

    return thr_t


def group_num_activation(
    l,
    group,
    threshold_dict={"unc": 0.001},
    results_dir="../results/number_baseline",
):
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    thr_method, thr_value = list(threshold_dict.items())[0]
    activations = []
    for number in numbers:
        activations.append(number_activation(l, number, group, threshold_dict))
    print(activations)
    # activations.plot(view="axial", colorbar=True)

    for ind, x in enumerate(activations):
        x.plot(view="mni", colorbar=True, **{"vmin": -8, "vmax": 8})
        plt.savefig(
            os.path.join(
                results_dir,
                f"{group['desc']}_{thr_method}_{thr_value}_{ind + 1}_ttest.png",
            )
        )

    return activations


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    [full_group, dd_group, ta_group] = groups
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    params = [{"unc": 0.001}, {"fdr": 0.05}, {"unc": 0.01}]
    # params = [{"unc": 0.001}]

    # for group in [full_group, dd_group, ta_group]:
    #     activations = []
    #     for number in numbers:
    #         for param in params:
    #             number_activation(l, number, group, param)

    group_num_activation(l, full_group, threshold_dict={"fdr": 0.05})
