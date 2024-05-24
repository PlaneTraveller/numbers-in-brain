#!/usr/bin/env python3

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from helpers import get_dd

import functools as ft

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


# =============================================================
# == Contrasts
def number_contrast(glm_result, contrast, results_dir="../results/subj_contrasts"):
    g1_dict = betas_ind(glm_result, contrast["g1"])
    g2_dict = betas_ind(glm_result, contrast["g2"])

    subj = glm_result["subj"]

    out = sum(g1_dict.values()) / len(contrast["g1"]) - sum(g2_dict.values()) / len(
        contrast["g2"]
    )
    out.plot()
    plt.savefig(
        os.path.join(results_dir, f"{subj}_" + contrast["name"] + "_contrast.png")
    )
    plt.close()
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
        # f"run-{run}_{contrast['name']}_contrasts.pkl",
        f"{group_desc}_run-{run}_{contrast['name']}_contrasts.pkl",
    )
    if os.path.exists(path_name):
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

    avg = sum(contrasts) / len(contrasts)
    avg.plot()
    plt.savefig(
        os.path.join(results_dir, f"{group_desc}_run-{run}_{contrast['name']}_avg.png")
    )

    contrasts = {"contrasts": contrasts, "avg": avg}

    with open(path_name, "wb") as f:
        pickle.dump(contrasts, f)

    return contrasts


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"
    run = 1
    proc_space = "MNI152NLin2009cAsym"

    # subj_list = ["0011", "0384", "0500", "0766"]

    even_odd = {
        "name": "odd_even",
        "g1": ["1", "3", "5", "7", "9"],
        "g2": ["2", "4", "6", "8"],
    }

    # even_odd_contrast = view_number_contrast(l, subj_list, run, even_odd)
    even_odd_contrast = ft.partial(view_number_contrast, l, run=1, contrast=even_odd)

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
    out = list(map(even_odd_contrast, groups))
