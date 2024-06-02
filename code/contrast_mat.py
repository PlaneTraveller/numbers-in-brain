#!/usr/bin/env python3
from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from helpers import get_dd
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

import bids
import matplotlib.pyplot as plt

from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix

# =============================================================
# == Contrasts
