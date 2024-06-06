#!/usr/bin/env python3

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
# == Creating Design Matrix
def events2dm(l, subj, task, run, dm_design):
    # dm_design: function that takes events df and returns a df with 3 columns: onset, duration, design
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
    # ['onset', 'duration', 'trial_type', 'response_time', 'correct', 'left_stim', 'right_stim', 'correct_response']

    dm_l = dm_design(events, "l")
    dm_l.columns = ["Onset", "Duration", "Stim"]

    dm_l_out = onsets_to_dm(dm_l, 1 / tr, n_tr)

    dm_r = dm_design(events, "r")
    dm_r.columns = ["Onset", "Duration", "Stim"]

    dm_r_out = onsets_to_dm(dm_r, 1 / tr, n_tr)
    # concatenate the two design matrices, dropping duplicate columns
    dm = pd.concat([dm_l_out, dm_r_out], axis=1)
    dm = dm.loc[:, ~dm.columns.duplicated()]

    # print(dm.info())
    # set the sampling frequency
    dm = Design_Matrix(dm, sampling_freq=1 / tr)
    return dm


def dm_convolve(l, subj, task, run, dm, hrf="canonical"):
    dm_conv = dm.convolve()
    return dm_conv


def dm_add_nuisance(l, subj, task, run, dm, duration=128, order=2):
    out = dm.add_dct_basis(duration=duration)
    out = out.add_poly(order=order, include_lower=True)
    return out


def dm_add_noise_cov(l, subj, task, run, dm):
    print(subj)
    noise_cov = l.get(
        scope="derivatives",
        datatype="func",
        subject=subj,
        task=task,
        run=run,
        desc="confounds",
        extension=".tsv",
    )[0]

    noise_cov = noise_cov.get_df()
    tr = l.get_tr()

    mc = noise_cov[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]
    mc_cov = make_motion_covariates(mc, tr)
    return pd.concat([dm, mc_cov], axis=1)


def make_motion_covariates(mc, tr):
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff() ** 2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1 / tr)


def dm_add_spikes(
    l, subj, task, run, dm, global_spike_cutoff=2.5, diff_spike_cutoff=2.5
):
    tr = l.get_tr()
    data_file = l.get(
        scope="derivatives",
        datatype="func",
        subject=subj,
        task=task,
        run=run,
        suffix="bold",
        extension=".nii.gz",
        return_type="file",
    )

    data = Brain_Data(data_file)
    spikes = data.find_spikes(
        global_spike_cutoff=global_spike_cutoff, diff_spike_cutoff=diff_spike_cutoff
    )

    spikes = Design_Matrix(spikes.iloc[:, 1:], sampling_freq=1 / tr)
    return pd.concat([dm, spikes], axis=1)


# =============================================================
# == Smoothen Data
def smooth_data(l, subj, task, run, fwhm=6, data_dir="../dataset/ds004791"):
    # Define the BIDS compliant file path for the smoothed data
    file_name = f"sub-{subj}_task-{task}_run-{run}_desc-smoothed_fwhm{fwhm}.nii.gz"
    json_file_name = f"sub-{subj}_task-{task}_run-{run}_desc-smoothed_fwhm{fwhm}.json"

    # Define the subject directory
    project_dir = data_dir + "/derivatives"
    subject_dir = os.path.join(project_dir, f"sub-{subj}", "func")

    # Attempt to load the smoothed data using BIDS loader
    try:
        smoothed_file = os.path.join(subject_dir, file_name)
        # print(smoothed_file)
        if smoothed_file:
            print(f"Loading existing smoothed data from {smoothed_file}")
            smoothed_img = nib.load(smoothed_file)
            smoothed_data = Brain_Data(smoothed_img)
            return smoothed_data
    except Exception as e:
        print(f"No existing smoothed data found. Proceeding with smoothing. Error: {e}")

    # Load the original data
    data_file = l.get(
        scope="derivatives",
        datatype="func",
        subject=subj,
        task=task,
        run=run,
        suffix="bold",
        extension=".nii.gz",
        return_type="file",
    )

    data = Brain_Data(data_file)

    # Perform smoothing
    smoothed_data = data.smooth(fwhm)

    os.makedirs(subject_dir, exist_ok=True)

    # Save the smoothed data as a NIfTI file
    smoothed_img = nib.Nifti1Image(
        smoothed_data.to_nifti().get_fdata(), smoothed_data.to_nifti().affine
    )
    smoothed_file_path = os.path.join(subject_dir, file_name)
    nib.save(smoothed_img, smoothed_file_path)
    print(f"Smoothed data saved to {smoothed_file_path}")

    # Create and save the sidecar JSON file with metadata
    sidecar_json = {
        "RepetitionTime": 2.0,  # Example TR, replace with actual value
        "TaskName": task,
        "Smooth": True,
        "FWHM": fwhm,
        "Description": "Smoothed fMRI data",
    }
    json_file_path = os.path.join(subject_dir, json_file_name)
    with open(json_file_path, "w") as f:
        json.dump(sidecar_json, f, indent=4)
    print(f"Sidecar JSON file saved to {json_file_path}")

    return smoothed_data


# =============================================================
# == Designs
def prelim_design(events):
    return events[["onset", "duration", "trial_type"]]


def find_number(stim):
    return re.search(r"\d", stim).group(0)


def numbers_design(events, param):
    # For the numbers task, concatenate the left and right stimuli; for the other tasks, use the trial_type
    if param == "l":
        direction = "left_stim"
    elif param == "r":
        direction = "right_stim"

    def row_proc(row):
        row = row[1]
        if row["trial_type"] == "NumberMatching":
            return "NumberMatching_" + param + "_" + find_number(row[direction])
        else:
            return row["trial_type"]

    stim = map(row_proc, events.iterrows())

    # concat stim with onset and duration
    return pd.concat([events[["onset", "duration"]], pd.Series(stim)], axis=1)


# =============================================================
# == GLM regression
def estimate_subj_glm(
    l,
    subj,
    task,
    run,
    dm_design=numbers_design,
    image_dir="../results/",
    betas_dir="../results/subj_glm",
):
    """
    Returns {subj, ind, stats}
    """
    # Define the file path for saving the GLM results
    file_name = f"sub-{subj}_task-{task}_run-{run}_glm_results.pkl"
    index_file_name = f"sub-{subj}_task-{task}_run-{run}_glm_index.txt"
    results_file_path = os.path.join(betas_dir, file_name)

    # Check if the GLM results file already exists
    if os.path.exists(results_file_path):
        print(f"Loading existing GLM results from {results_file_path}")
        with open(results_file_path, "rb") as f:
            stats = pickle.load(f)
        with open(os.path.join(betas_dir, index_file_name), "r") as f:
            ind = [line.strip() for line in f]
        return {"stats": stats, "index": ind, "subj": subj}

    # Define the operations for the design matrix
    dm_op = [
        events2dm,
        dm_convolve,
        dm_add_nuisance,
        dm_add_noise_cov,
        dm_add_spikes,
    ]
    dm_op_curried = [ft.partial(f, l, subj, task, run) for f in dm_op]
    final_dm = ft.reduce(lambda x, y: y(x), dm_op_curried, dm_design)

    # Save design matrix heatmap
    final_dm.heatmap(cmap="RdBu_r", vmin=-1, vmax=1)
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(
        os.path.join(image_dir, f"sub-{subj}_task-{task}_run-{run}_final_dm.png")
    )
    plt.close()

    # Perform smoothing
    data = smooth_data(l, subj, task, run)
    data.X = final_dm
    ind = data.X.columns

    # Perform GLM regression
    stats = data.regress()

    # Create the results directory if it doesn't exist
    os.makedirs(betas_dir, exist_ok=True)

    # Save the GLM results to a file
    with open(results_file_path, "wb") as f:
        pickle.dump(stats, f)

    with open(os.path.join(betas_dir, index_file_name), "w") as f:
        for item in ind:
            f.write(f"{item}\n")
    print(f"GLM results saved to {results_file_path}")

    return {"stats": stats, "index": ind, "subj": subj}


# def betas_ind(glm_result, numbers, main_task="NumberMatching"):
#     ind = glm_result["index"]
#     # print(ind)
#     betas = glm_result["stats"]["beta"]

#     # construct dict from ind and betas
#     betas_dict = dict(zip(ind, betas))

#     def contain_numbers(s):
#         try:
#             task_correct = s.split("_")[0] == main_task
#             # print(task_correct)
#             num_correct = find_number(s) in numbers
#             return task_correct and num_correct
#         except:
#             return False

#     out = {k: v for k, v in betas_dict.items() if contain_numbers(k)}

#     return out


def glm_ind(glm_result, condition, category):
    ind = glm_result["index"]
    # print(ind)
    data = glm_result["stats"][category]

    # construct dict from ind and betas
    data_dict = dict(zip(ind, data))

    def meet_condition(s):
        try:
            # task_correct = s.split("_")[0] == main_task
            # # print(task_correct)
            # num_correct = find_number(s) in numbers
            return condition(s)
        except:
            return False

    out = {k: v for k, v in data_dict.items() if meet_condition(k)}

    return out


def number_cond(numbers, main_task="NumberMatching"):
    def condition(s):
        try:
            task_correct = s.split("_")[0] == main_task
            # print(task_correct)
            num_correct = find_number(s) in numbers
            return task_correct and num_correct
        except:
            return False

    return condition


def shape_cond(s):
    try:
        return s.split("_")[0] == "ShapeMatching"
    except:
        return False


def betas_ind(glm_result, numbers, main_task="NumberMatching"):
    return glm_ind(glm_result, number_cond(numbers, main_task=main_task), "beta")


def shape_ind(glm_result):
    return glm_ind(glm_result, shape_cond, "beta")


def p_ind(glm_result, numbers, main_task="NumberMatching"):
    return glm_ind(glm_result, number_cond(numbers, main_task=main_task), "p")


def subj_p_ind(glm_result, subj):
    return glm_ind(glm_result, lambda s: s.split("_")[-1] == subj, "p")


# =============================================================
# == Run
if __name__ == "__main__":
    data_dir = "../dataset/ds004791"
    results_dir = "../results"

    l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

    task = "matching"
    run = 1
    proc_space = "MNI152NLin2009cAsym"

    test_subj = "0384"
    # subj_list = ["0011", "0384", "0500", "0766"]

    get_glm = ft.partial(estimate_subj_glm, l, task=task, run=run)
    glm_stats = get_glm(test_subj)["stats"]

    glm_stats["p"][2].plot(
        view="mni", colorbar=True, threshold_upper=0, threshold_lower=0.001
    )
    plt.savefig(os.path.join(results_dir, "try", "glm_p30.png"))
    plt.close()
