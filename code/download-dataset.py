#!/usr/bin/env python3


import os
import os.path as op

import openneuro as on

from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

dataset_id = "ds004791"

bids_root = op.join("../dataset", dataset_id)

if not op.isdir(bids_root):
    os.makedirs(bids_root)

on.download(dataset=dataset_id, target_dir=bids_root)
