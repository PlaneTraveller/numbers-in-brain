#!/usr/bin/env python3

from matplotlib import pyplot as plt

# mcr_xvals = [0.58, 0.54, 0.5, 0.58, 0.5, 0.56, 0.48, 0.56, 0.48, 0.48]

pvals = [
    0.11488511488511488,
    0.4905094905094905,
    0.36163836163836166,
    0.1918081918081918,
    0.2017982017982018,
    0.1108891108891109,
    0.08091908091908091,
    0.23476523476523475,
    0.4095904095904096,
    0.43656343656343655,
]

cr = [
    0.6428571428571429,
    0.5612244897959183,
    0.5765306122448981,
    0.6198979591836735,
    0.6173469387755102,
    0.6403061224489797,
    0.6607142857142857,
    0.6173469387755102,
    0.5637755102040816,
    0.5561224489795917,
]

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)

print(sum(cr) / len(cr))

bp = ax.boxplot(
    cr,
    showmeans=True,
    vert=True,
    meanline=True,
    labels=[""],
)

ax.set_title("SVM CV Scores")
ax.set_ylabel("Prediction Accuracy")

plt.savefig("svm_plot.png")
