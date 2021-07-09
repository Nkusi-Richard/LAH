#!/usr/bin/env python3
"""
Script to plot the confusion matrix

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(conf_mat, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.

    Args:
        - cm (np.array): matrix

        - classes (int): number of classes

        -normalize (bool): to apply a normalization on cm

        - title (str): title of the plot

        - cmap (plt.cm): to specify a plot cmap

    Returns:
        - axe (plt.axes): Axes object

    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        np.seterr(divide='ignore', invalid='ignore')
        conf_mat = conf_mat.astype(
            'float') / conf_mat.sum(axis=1)[:, np.newaxis]

    fig, axe = plt.subplots()
    img = axe.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    axe.figure.colorbar(img)
    """axe.set(xticks=np.arange(conf_mat.shape[1]),
            yticks=np.arange(conf_mat.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')"""

    plt.xticks(range(conf_mat.shape[1]), classes[:conf_mat.shape[1]])
    plt.yticks(range(conf_mat.shape[0]), classes[:conf_mat.shape[0]])
    axe.set_xticklabels((axe.get_xticks() +1).astype(str))
    axe.set_yticklabels((axe.get_yticks() +1).astype(str))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(axe.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    plot_cm = np.round(conf_mat, decimals=2)
    fmt = 'float' if normalize else 'int'
    B = conf_mat[~np.isnan(conf_mat)]
    thresh = B.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            # print(conf_mat[i, j])
            axe.text(j, i, plot_cm[i, j].astype(fmt),
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    fig.tight_layout()
    return axe
