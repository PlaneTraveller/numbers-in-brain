#!/usr/bin/env python3

import bids
from nilearn import image as nimg
from nilearn import plotting as nplot
import seaborn as sns

# from bids import BIDSValidator
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from nltools.file_reader import onsets_to_dm
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import find_spikes
from nilearn.plotting import view_img, glass_brain, plot_stat_map

import nibabel as nib

data_dir = "../dataset/ds004791"
# processed_dir = "../dataset/out"
img_dir = "./img"

# =============================================================
# == Loading data
# l = bids.BIDSLayout(data_dir)
l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

test_sub = "0011"
test_task = "matching"
test_run = 1
proc_space = "MNI152NLin2009cAsym"


def play():
    func_data = l.get(
        datatype="func",
        subject=test_sub,
        task=test_task,
        run=test_run,
        desc="preproc",
        extension=".nii.gz",
    )

    # Can use suffix to get events
    events_data = l.get(
        datatype="func", subject=test_sub, task=test_task, run=test_run, suffix="events"
    )[0]

    events_df = events_data.get_df()

    print(events_df.to_string())

    func_img = nimg.load_img(func_data)
    print(func_img.shape)
    # (84, 84, 48, 309)
    # preprocessed: (63, 77, 66, 309)

    func_vol5 = func_img.slicer[:, :, :, 4]
    view = nplot.view_img(func_vol5, threshold=3)
    view.open_in_browser()


# play()

def events2dm(l, subj, task, run):
    tr = l.get_tr()
    data_file = l.get(
        scope="raw",
        datatype="func",
        subject=subj,
        task=task,
        run=run,
        suffix="bold",
        extension=".nii.gz",
        return_type="file",
    )[0]
    data = nib.load(data_file)
    n_tr = data.shape[-1]
    # print(n_tr)

    events_file = l.get(
        scope="raw",
        datatype="func",
        subject=subj,
        task=task,
        run=run,
        suffix="events",
        extension=".tsv",
    )[0]

    events = events_file.get_df()
    # print(events.columns)
    # ['onset', 'duration', 'trial_type', 'response_time', 'correct', 'left_stim', 'right_stim', 'correct_response']
    dm_design = events[["onset", "duration", "trial_type"]]
    dm_design.columns = ["Onset", "Duration", "Stim"]
    return onsets_to_dm(dm_design, 1 / tr, n_tr)

# =============================================================
# == Data indexing playground

image_dir = "./img/"
test_sub = "0011"
test_task = "matching"
test_run = 1
proc_space = "MNI152NLin2009cAsym"

test_dm = events2dm(l, "0011", "matching", 1)
print(test_dm.head())

f, a = plt.subplots(figsize=(10, 5))

# Design matrix
test_dm.plot(ax=a)

fname = image_dir + 'design_mat.png'
plt.savefig(fname)
plt.close()


# Heatmap
test_dm.heatmap()
plt.savefig(image_dir + 'design_heatmap.png')
plt.close()

# Heatmap after HRFconvolution
test_dm_convolve = test_dm.convolve()
test_dm_convolve.heatmap()
plt.savefig(image_dir + 'design_heatmap_conv.png')
plt.close()

f, a = plt.subplots(figsize=(10, 5))
test_dm_convolve.plot(ax=a)
plt.savefig(image_dir + 'design_conv.png')
plt.close()

dm_conv_filt = test_dm_convolve.add_dct_basis(duration=128)
# print(dm_conv_filt)
dm_conv_filt.iloc[:,3:].plot()

plt.savefig(image_dir + 'dm_filt.png')
plt.close()

dm_conv_filt.heatmap()
plt.savefig(image_dir + 'dm_filt_heat.png')
plt.close()

dm_conv_filt_poly = dm_conv_filt.add_poly(order=2, include_lower=True)
# print(dm_conv_filt_poly.head())
dm_conv_filt_poly.heatmap()
plt.savefig(image_dir + 'dm_poly.png')
plt.close()

subj = "0011"
task = "matching"
run = 1
covariates = l.get(
    scope="derivatives",
    datatype="func",
    subject=subj,
    task=task,
    run=run,
    desc="confounds",
    extension="tsv",
)[0].get_df()

mc = covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]

plt.figure(figsize=(15, 5))
plt.plot(zscore(mc))
plt.savefig(image_dir + 'motion_corr')
plt.close()

def make_motion_covariates(mc, tr):
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1/tr)

tr = l.get_tr()
mc_cov = make_motion_covariates(mc, tr)

sns.heatmap(mc_cov)
plt.savefig(image_dir + 'mc_cov')
plt.close()

test_data = Brain_Data(l.get(
    scope="derivatives",
    datatype="func",
    subject=subj,
    task=task,
    run=run,
    suffix='bold',
    extension='nii.gz',
    return_type='file'
))

plt.figure(figsize=(15,3))
plt.plot(np.mean(test_data.data, axis=1), linewidth=3)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Intensity', fontsize=18)

plt.savefig(image_dir + 'sig_intensity')
plt.close()

spikes = test_data.find_spikes(global_spike_cutoff=2.5, diff_spike_cutoff=2.5)

f, a = plt.subplots(figsize=(15,3))
spikes = Design_Matrix(spikes.iloc[:,1:], sampling_freq=1/tr)
spikes.plot(ax=a, linewidth=2)

plt.savefig(image_dir + 'spikes')
plt.close()

final_dm = pd.concat([dm_conv_filt_poly, mc_cov, spikes], axis=1)
final_dm.heatmap(cmap='RdBu_r', vmin=-1, vmax=1)

plt.savefig(image_dir + 'final_dm')
plt.close()

fwhm = 6
smoothed = test_data.smooth(fwhm=fwhm)

test_data.mean().plot()

plt.savefig(image_dir + 'original_mean.png')
plt.close()

smoothed.mean().plot()
plt.savefig(image_dir + 'smoothed_mean.png')
plt.close()

smoothed.X = final_dm
stats = smoothed.regress()
print(stats.keys())

print(smoothed.X.columns)


