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
from classifier import load_data
from helpers import get_dd

# from nltools.stats import one_sample_ttest
# from statsmodels.stats.multitest import multipletests


def extract_betas_by_label(data, label):
    mask = data.Y == label
    return data[mask]


def number_ttest(
    data,
    group_desc,
    number,
    p_value=0.05,
    th_param={"fdr": 0.05},
    results_dir="../results",
):
    extracted = extract_betas_by_label(data, number)
    th_method = list(th_param.keys())[0]
    th_value = th_param[th_method]

    res = extracted.ttest(threshold_dict=th_param)
    # print(res)
    # p = res["p"]
    # p.plot(view="mni", colorbar=True, threshold_upper=0, threshold_lower=p_value)
    # p.plot(view="mni", colorbar=True)

    thr_t = res["thr_t"]
    thr_t.plot(view="mni", colorbar=True)
    plt.savefig(
        os.path.join(
            results_dir,
            "activation",
            # f"{group_desc}_p{p_value}_{th_method}_{th_value}_{number}_ttest.png",
            f"{group_desc}_{th_method}_{th_value}_{number}_ttest.png",
        )
    )
    plt.close()


def activation_map(
    l, group, p_value=0.05, th_param={"fdr": 0.05}, results_dir="../results"
):
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    data = load_data(l, group)
    group_desc = group["desc"]
    number_ttest_curried = ft.partial(
        number_ttest,
        data,
        group_desc,
        p_value=p_value,
        th_param=th_param,
        results_dir=results_dir,
    )

    list(map(number_ttest_curried, numbers))


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

    # # Uncorrected: threshold for any voxel as we are not controlling for multiple tests
    # activation_map(l, full_group, th_param={"unc": 0.001})
    activation_map(l, dd_group, th_param={"unc": 0.01})
    activation_map(l, ta_group, th_param={"unc": 0.01})

    # # FDR corrected
    # activation_map(l, full_group, th_param={"fdr": 0.05})

    # activation_map(l, full_group, th_param={"unc": 0.005})
    # activation_map(l, full_group, th_param={"unc": 0.01})
    # activation_map(l, full_group, th_param={"fdr": 0.1})
