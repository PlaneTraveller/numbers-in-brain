#!/usr/bin/env python3

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from subject_glm import find_number
from helpers import get_dd

import functools as ft
from itertools import chain

# import numpy as np
import pandas as pd
import re
import os
import json
import pickle

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
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================
# == Noting down
# get_run1_glm = ft.partial(get_glm, l, task=task, run=1)
# get_run2_glm = ft.partial(get_glm, l, task=task, run=2)

# # 259 GB - 49.8 GB
# errorList = []
# for subj in subjs:
#     try:
#         glm = get_run1_glm(subj)
#     except:
#         errorList.append(subj)
#         continue

# for subj in subjs:
#     try:
#         glm = get_run2_glm(subj)
#     except:
#         errorList.append(subj)
#         continue

# print(errorList)
# data = ft.reduce(ft.partial(load_subj_beta, l), subjs, ([], []))
# print(data)


# =============================================================
# == Constructing dataset
def load_subj_glm(l, subj):
    """
    Returns a tuple. First element is {"subj": [], "run": [], "stim": []}, second element is betas.
    """
    task = "matching"
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print(subj)

    get_run1_glm = ft.partial(get_glm, l, task=task, run=1)
    get_run2_glm = ft.partial(get_glm, l, task=task, run=2)
    glm1 = get_run1_glm(subj)
    glm2 = get_run2_glm(subj)
    betas_ind1 = betas_ind(glm1, numbers)
    betas_ind2 = betas_ind(glm2, numbers)

    def parse_one(result_tup, data_tup):
        return (
            {
                "subj": result_tup[0]["subj"] + [subj],
                "stim": result_tup[0]["stim"] + [find_number(data_tup[0])],
            },
            result_tup[1] + [data_tup[1]],
        )

    out1 = ft.reduce(parse_one, betas_ind1.items(), ({"subj": [], "stim": []}, []))
    out1[0]["run"] = ["1"] * len(out1[0]["stim"])
    out2 = ft.reduce(parse_one, betas_ind2.items(), ({"subj": [], "stim": []}, []))
    out2[0]["run"] = ["2"] * len(out2[0]["stim"])

    out = (
        {
            "stim": out1[0]["stim"] + out2[0]["stim"],
            "subj": out1[0]["subj"] + out2[0]["subj"],
            "run": out1[0]["run"] + out2[0]["run"],
        },
        out1[1] + out2[1],
    )
    print("Loaded betas for", subj)
    return out


def load_subj_labels_beta(l, subj, tup):
    """
    Takes a subject and a tuple of label-beta list pairs and
    returns such a tuple of betas appended with the subject's betas.

    Strips the label to be the number preceived by the subject.
    """
    labels, betas = load_subj_glm(l, subj)
    labels = list(map(find_number, labels["stim"]))
    out = (
        {
            "subj": tup[0]["subj"] + labels["subj"],
            "stim": tup[0]["stim"] + labels["stim"],
            "run": tup[0]["run"] + labels["run"],
        },
        tup[1] + betas,
    )
    return out


def load_data(l, group, results_dir="../results"):
    selection_desc = group["desc"]
    file_name = selection_desc + "_classifier-data.pkl"
    path_name = os.path.join(results_dir, file_name)
    subjs = group["subjs"]

    if os.path.exists(path_name):
        with open(path_name, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        labels, data = ft.reduce(ft.partial(load_subj_labels_beta, l), subjs, ({}, []))
        data = Brain_Data(data, Y=pd.DataFrame(labels))
        with open(path_name, "wb") as f:
            pickle.dump(data, f)
        return data


def classify(l, group, algorithm, algo_args, n_fold=3, results_dir="../results"):
    # print(group)
    group_desc = group["desc"]

    path_name = os.path.join(
        results_dir,
        group_desc + "_" + algorithm + "_" + str(n_fold) + "-fold_classifier-stats.pkl",
    )
    if os.path.exists(path_name):
        print("Loading stats for", group_desc)
        with open(path_name, "rb") as f:
            stats = pickle.load(f)
        return stats

    data = load_data(l, group)
    stats = data.predict(
        algorithm=algorithm,
        cv_dict={"type": "kfolds", "n_folds": n_fold, "n": len(data.Y)},
        plot=False,
        **algo_args,
    )
    # Save stats
    with open(path_name, "wb") as f:
        print("Saving stats for", group_desc)
        pickle.dump(stats, f)
    return stats


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"

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
    linear_classifier = ft.partial(
        classify, l, algorithm="svm", algo_args={"kernel": "linear"}
    )
    rbf_classifier = ft.partial(
        classify, l, algorithm="svm", algo_args={"kernel": "rbf"}
    )
    # poly_classifier = ft.partial(
    #     classify, l, algorithm="svm", algo_args={"kernel": "poly"}
    # )
    out1 = list(map(linear_classifier, groups))
    out2 = list(map(rbf_classifier, groups))
    print(out1)
    print(out2)

    # full: 0.12
    # dd: 0.11
    # ta: 0.14
