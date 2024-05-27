#!/usr/bin/env python3

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

# from classifier import load_data
from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from subject_glm import glm_ind
from subject_glm import find_number
from subject_glm import betas_ind, p_ind
from subject_glm import subj_p_ind
from classifier import load_subj_glm
from helpers import get_dd

# from nltools.stats import one_sample_ttest
# from statsmodels.stats.multitest import multipletests


# =============================================================
# == Constructing DM
def collect_subject(l, tup, subj):
    labels, betas = load_subj_glm(l, subj)
    out = {
        "subj": tup[0]["subj"] + labels["subj"],
        "stim": tup[0]["stim"] + labels["stim"],
        "run": tup[0]["run"] + labels["run"],
    }

    return (out, tup[1] + betas)


def create_dm(l, group, results_dir="../results/group_glm"):
    """
    Create a design matrix: subjects + conditions + runs
    Used in pair with load_data
    """
    subjs = group["subjs"]

    parse_one = ft.partial(collect_subject, l)

    dm, betas = ft.reduce(parse_one, subjs, ({"subj": [], "stim": [], "run": []}, []))

    dm = pd.DataFrame(dm)
    dm = pd.get_dummies(dm, columns=["stim", "run", "subj"])
    dm = Design_Matrix(dm)
    dm.info()
    dm.heatmap(cmap="RdBu_r", vmin=-1, vmax=1)
    plt.savefig(os.path.join(results_dir, f"design_matrix_{group['desc']}.png"))

    data = Brain_Data(data=betas, X=dm)
    return data
    return dm.columns


def estimate_group_glm(l, group, results_dir="../results/group_glm"):
    path_name = os.path.join(results_dir, group["desc"] + "_glm.pkl")
    ind_path_name = os.path.join(results_dir, group["desc"] + "_ind.txt")

    if os.path.exists(path_name):
        with open(path_name, "rb") as f:
            stats = pickle.load(f)
        with open(ind_path_name, "r") as f:
            ind = [line.strip() for line in f]
        return {"stats": stats, "index": ind, "group": group["desc"]}
    else:
        data = create_dm(l, group)
        ind = data.X.columns
        stats = data.regress()
        with open(path_name, "wb") as f:
            pickle.dump(stats, f)
        with open(ind_path_name, "w") as f:
            for line in ind:
                f.write(line + "\n")

        return {"stats": stats, "index": ind, "group": group["desc"]}


def two_sample_ttest(l, g1, g2, contrast, results_dir="../results/two_ttest"):
    pass


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"
    numbers = ["1", "2", "3", "4", "6", "7", "8", "9"]

    subjs = l.get_subjects()
    error_subjs = ["0192", "0487", "1085"]

    subjs = [subj for subj in subjs if subj not in error_subjs]

    toy_group = {"desc": "toy", "subjs": subjs[:5]}
    toy10_group = {"desc": "toy-10", "subjs": subjs[:10]}
    full_group = {"desc": "full", "subjs": subjs}

    # glm = estimate_group_glm(l, full_group)

    # activation_map = betas_ind(glm, numbers, "stim")
    # activation_map["stim_9"].plot()
    # plt.savefig(os.path.join(results_dir, "group_glm", "group_glm_stim_9.png"))

    # p2 = p_ind(glm, "2", "stim")
    # list(p2.values())[0].plot(
    #     cmap="RdBu_r", colorbar=True, threshold_upper=0, threshold_lower=0.05
    # )
    # plt.savefig(os.path.join(results_dir, "group_glm", "group_glm_stim_2_p.png"))

    # run1_betas = list(betas_ind(glm, "1", "run").values())[0]
    # run1_p = list(p_ind(glm, "1", "run").values())[0]

    # run1_p.plot(cmap="RdBu_r", colorbar=True)
    # plt.savefig(os.path.join(results_dir, "group_glm", "group_glm_run_1_p.png"))

    # subj = "0011"
    # subj_p = list(subj_p_ind(glm, subj).values())[0]

    # subj_p.plot(cmap="RdBu_r", colorbar=True)
    # plt.savefig(os.path.join(results_dir, "group_glm", f"group_glm_{subj}_p.png"))

    dd_ta = get_dd(full_group)
    dd_group = {"desc": "DD_full", "subjs": dd_ta["DD"]}
    ta_group = {"desc": "TA_full", "subjs": dd_ta["TA"]}

    glm_ta = estimate_group_glm(l, ta_group)

    # p2 = p_ind(glm_ta, "2", "stim")
    # list(p2.values())[0].plot(
    #     cmap="RdBu_r", colorbar=True, threshold_upper=0, threshold_lower=0.05
    # )
    # plt.savefig(os.path.join(results_dir, "group_glm", "ta_glm_stim_2_p.png"))

    run1 = p_ind(glm_ta, "1", "run")
    run1_p = list(run1.values())[0]
    run1_p.plot(cmap="RdBu_r", colorbar=True)
    plt.savefig(os.path.join(results_dir, "group_glm", "group_glm_run_1_p.png"))
