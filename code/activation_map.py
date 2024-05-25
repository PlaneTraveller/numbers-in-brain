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

# from nltools.stats import one_sample_ttest
# from statsmodels.stats.multitest import multipletests


def extract_betas_by_label(data, label):
    mask = data.Y == label
    return data[mask]


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

    data = load_data(l, toy_group)

    beta_3 = extract_betas_by_label(data, "3")

    res = beta_3.ttest()
    p = res["p"]
    p.plot(view="mni", colorbar=True, threshold_upper=0, threshold_lower=0.05)
    plt.savefig(os.path.join(results_dir, "activation", "toy_group_3_ttest.png"))
