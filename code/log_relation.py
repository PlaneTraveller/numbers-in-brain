#!/usr/bin/env python3

#!/usr/bin/env python3

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from helpers import get_dd
import helpers as h
from math import log

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
# == Code


def load_num_betas(group, num):
    subjs = group["subjs"]

    glm_1 = ft.partial(get_glm, l, run=1)
    glm_2 = ft.partial(get_glm, l, run=2)

    subj = glm_result["subj"]
    betas = map(lambda x: betas_ind(glm_result, [x]), numbers)


def pair_num_distance(group, results_dir="../results/log_relation/"):
    numbers = [
        "1",
    ]
    group_desc = group["desc"]

    group_betas = 1

    return


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"
    run = 1
    proc_space = "MNI152NLin2009cAsym"
