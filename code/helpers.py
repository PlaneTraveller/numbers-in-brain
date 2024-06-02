#!/usr/bin/env python3

import os
import pandas as pd
import functools as ft
import bids
from math import log


def get_methods(object, spacing=20):
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: " ".join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(
                str(method.ljust(spacing))
                + " "
                + processFunc(str(getattr(object, method).__doc__)[0:90])
            )
        except Exception:
            print(method.ljust(spacing) + " " + " getattr() failed")


def get_dd(group, data_dir="../dataset/ds004791"):
    with open(os.path.join(data_dir, "participants.tsv"), "r") as f:
        data = pd.read_csv(f, sep="\t")

    data = dict(zip(data["participant_id"], data["group"]))
    # print(data)

    def which_group(out, subj):
        # print(out)
        # print(data["sub-" + subj])
        if data["sub-" + subj] == "DD":
            return {"DD": out["DD"] + [subj], "TA": out["TA"]}
        else:
            return {"DD": out["DD"], "TA": out["TA"] + [subj]}

    out = ft.reduce(which_group, group["subjs"], {"DD": [], "TA": []})
    return out


# def load_data(subjs, stims, runs):

#     return


# ===========
# == Other Stuff
data_dir = "../dataset/ds004791"
results_dir = "../results"

l = bids.BIDSLayout(data_dir, derivatives=True, config=["bids", "derivatives"])

task = "matching"
run = 1
proc_space = "MNI152NLin2009cAsym"

subjs = l.get_subjects()
error_subjs = ["0192", "0487", "1085"]

subjs = [subj for subj in subjs if subj not in error_subjs]

toy_group = {"desc": "toy", "subjs": subjs[:5]}
toy10_group = {"desc": "toy-10", "subjs": subjs[:10]}
full_group = {"desc": "full", "subjs": subjs}

dd_ta = get_dd(full_group)
dd_group = {"desc": "DD_full", "subjs": dd_ta["DD"]}
ta_group = {"desc": "TA_full", "subjs": dd_ta["TA"]}

groups = [full_group, dd_group, ta_group]

even_odd = {
    "name": "even_odd",
    "weight": [1, -1, 1, -1, 1, -1, 1, -1, 1],
}

linear_relation = {
    "name": "linear_relation",
    "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
}

new_linear_relation = {
    "name": "new_linear_relation",
    "weight": list(map(lambda x: 1 / x, [1, 2, 3, 4, 5, 6, 7, 8, 9])),
}

rev_linear_relation = {
    "name": "reverse_linear_relation",
    "weight": [9, 8, 7, 6, 5, 4, 3, 2, 1],
}

log_relation = {
    "name": "log_relation",
    "weight": list(map(log, [1, 2, 3, 4, 5, 6, 7, 8, 9])),
}

new_log_relation = {
    "name": "new_log_relation",
    "weight": [0.01] + list(map(lambda x: 1 / log(x), [2, 3, 4, 5, 6, 7, 8, 9])),
}

num_1 = {
    "name": "num_1",
    "weight": [1, 0, 0, 0, 0, 0, 0, 0, 0],
}
num_2 = {
    "name": "num_2",
    "weight": [0, 1, 0, 0, 0, 0, 0, 0, 0],
}
num_3 = {
    "name": "num_3",
    "weight": [0, 0, 1, 0, 0, 0, 0, 0, 0],
}
num_4 = {
    "name": "num_4",
    "weight": [0, 0, 0, 1, 0, 0, 0, 0, 0],
}
num_5 = {
    "name": "num_5",
    "weight": [0, 0, 0, 0, 1, 0, 0, 0, 0],
}
num_6 = {
    "name": "num_6",
    "weight": [0, 0, 0, 0, 0, 1, 0, 0, 0],
}
num_7 = {
    "name": "num_7",
    "weight": [0, 0, 0, 0, 0, 0, 1, 0, 0],
}
num_8 = {
    "name": "num_8",
    "weight": [0, 0, 0, 0, 0, 0, 0, 1, 0],
}
num_9 = {
    "name": "num_9",
    "weight": [0, 0, 0, 0, 0, 0, 0, 0, 1],
}
