#!/usr/bin/env python3

from matplotlib import pyplot as plt

mcr_xvals = [0.58, 0.54, 0.5, 0.58, 0.5, 0.56, 0.48, 0.56, 0.48, 0.48]

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)


bp = ax.boxplot(
    mcr_xvals,
    showmeans=True,
    vert=True,
    meanline=True,
    labels=[""],
)

ax.set_title("SVM CV Scores")
ax.set_ylabel("Prediction Accuracy")

plt.savefig("svm_plot.png")
