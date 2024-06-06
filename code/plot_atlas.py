#!/usr/bin/env python3

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from helpers import get_dd
from math import log
from random import sample
from random import seed
from os import makedirs

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

mask = Brain_Data("../dataset/k50_2mm.nii.gz")
mask.plot(colorbar=True)
plt.savefig("mask.png")
