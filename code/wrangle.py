#!/usr/bin/env python3

import bids
from nilearn import image as nimg
from nilearn import plotting as nplot
import matplotlib.pyplot as plt
import numpy as np

data_dir = "../dataset/ds004791"
img_dir = "./img"

#=============================================================
#== Loading data
l = bids.BIDSLayout(data_dir)
sub = "0011"
task = "matching"
run = 1

anat_data = l.get(datatype='anat', subject = sub, extension=".nii.gz")
func_data = l.get(subject = sub, task=task, run=run, extension=".nii.gz")

func_file = l.get(subject = sub, task=task, run=run, extension=".nii.gz", return_type='file')
print(func_file)

# print(l.get_entities()['space'])

func_img = nimg.load_img(func_file)
# print(func_img.shape)
# (84, 84, 48, 309)

func_vol5 = func_img.slicer[:, :, :, 4]
view = nplot.view_img(func_vol5, threshold=3)
view.open_in_browser()
