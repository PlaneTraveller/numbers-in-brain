#!/usr/bin/env python3

import functools as ft

import pandas as pd
import re
import os
import pickle
import json

import bids

import matplotlib.pyplot as plt

import nibabel as nib
from nltools import Brain_Data
from nilearn import image, datasets, plotting
from nilearn.regions import connected_label_regions
from nilearn.image import resample_to_img
from nltools.mask import expand_mask
import numpy as np

from subject_glm import estimate_subj_glm as get_glm
from subject_glm import betas_ind
from subject_glm import glm_ind
from subject_glm import find_number
from subject_glm import betas_ind, p_ind
from subject_glm import subj_p_ind
from classifier import load_subj_glm
from classifier import load_data
from helpers import get_dd
from activation_map import activation_map


# =============================================================
# == Prepare parcellation atlas
def get_parcellation(results_dir="../results/roi"):
    yeo = datasets.fetch_atlas_yeo_2011()

    for label in ["thick_7", "thick_17"]:
        n = label.replace("thick_", "")
        atlas = yeo[label]  # this loads in a .nii file
        # plotting.plot_roi(atlas, title=f"Yeo - {n} Network", cmap="Paired")

    parcellation = connected_label_regions(atlas)

    plotting.plot_roi(parcellation, title="Yeo", colorbar=True, cmap="Paired")
    plt.savefig(os.path.join(results_dir, "yeo.png"))
    plt.close()

    return parcellation


def get_roi_mask(thresholded_maps, parcellation, results_dir="../results/roi"):
    reference_img = thresholded_maps[0].to_nifti()
    parcellation_resampled = resample_to_img(
        source_img=parcellation,
        target_img=reference_img,
        interpolation="nearest",
    )
    atlas_data = parcellation_resampled.get_fdata()

    # print(atlas_data.shape)
    # 256, 256, 256
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude background

    # Identify significant parcels
    significant_parcels = set()
    for thresholded_map in thresholded_maps:
        thresholded_data = thresholded_map.to_nifti().get_fdata()
        # print(thresholded_data.shape)
        # (91, 109, 91)
        significant_voxels = np.where(thresholded_data > 0)
        for x, y, z in zip(*significant_voxels):
            # print(x, y, z)
            parcel_id = atlas_data[x, y, z]
            if parcel_id > 0:  # Exclude background
                significant_parcels.add(parcel_id)

    # Create an empty mask
    combined_mask_data = np.zeros(atlas_data.shape)
    for parcel_id in significant_parcels:
        combined_mask_data[atlas_data == parcel_id] = 1

    # Convert to NIfTI image
    combined_mask_img = image.new_img_like(parcellation, combined_mask_data)

    # Save the combined ROI mask
    combined_mask_path = os.path.join(results_dir, "combined_rois.nii.gz")
    combined_mask_img.to_filename(combined_mask_path)

    # Plot and save the combined ROI mask
    plotting.plot_roi(combined_mask_img, title="Combined ROIs")
    plt.savefig(os.path.join(results_dir, "combined_rois.png"))
    plt.close()

    return Brain_Data(combined_mask_img)


def get_roi_labels(activations, parcellation, results_dir="../results/roi"):
    parcellation = Brain_Data(parcellation)
    empty_mask = Brain_Data(np.zeros(parcellation.shape))

    parcellation.plot()
    plt.savefig(os.path.join(results_dir, "parcellation.png"))

    mask = expand_mask(parcellation)

    activations = list(map(lambda x: x.apply_mask(mask), activations))

    accum_activations = sum(activations)

    accum_activations.plot()
    plt.savefig(os.path.join(results_dir, "accum_activations.png"))

    # return all indeces of the mask where dot product is not zero


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

    activation = activation_map(l, full_group, th_param={"unc": 0.001})
    parcellation = get_parcellation()

    get_roi_labels(activation, parcellation)
