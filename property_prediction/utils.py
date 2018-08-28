'''
File: utils.py
Project: ML_workshop
File Created: Saturday, 11th August 2018 6:44:34 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
'''

import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Returns the root-mean-squared-error

    Arguments:
        y_true {array-like} -- true target values
        y_pred {array-like} -- predicted values

    Returns:
        float -- rmse
    """
    return sqrt(mean_squared_error(y_true, y_pred))


def plot_predictions(y, y_hat, labels, save_path=None):
    """Scatter plots for prediction vs true values

    Arguments:
        y {list of array-like} -- list of targets
        y_hat {list of array-like} -- list of predictions
        labels {list of str} -- list of labels

    Keyword Arguments:
        save_path {str} -- path to save figure to (default: {None})
    """
    matplotlib.rcParams.update({'font.size': 22})
    y, y_hat, labels = list(
        map(
            lambda l: l if isinstance(l, list) else [l],
            [y, y_hat, labels]))
    n_plots = len(y)
    y_min = min([min(z) for z in y])
    y_max = max([max(z) for z in y])
    lims = (y_min, y_max)
    fig, ax = plt.subplots(
        1, n_plots, figsize=(10*n_plots, 8),
        squeeze=False, sharex=True, sharey=True)
    for axis, target, prediction, label in zip(ax[0, :], y, y_hat, labels):
        # Scatter plot
        axis.scatter(target, prediction, alpha=0.3)

        # Title and labels
        rmse_value = rmse(target, prediction)
        title = label + " (RMSE=%.3f)" % rmse_value
        axis.set_title(title)
        axis.set_xlabel('Target Compressive Strength (MPa)')
        axis.set_ylabel('Predicted Compressive Strength (MPa)')
        axis.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)


def plot_feature_importances(importances, columns, save_path=None):
    """plot feature importances

    Arguments:
        importances {list of float} -- feature importances
        columns {list of str} -- labels

    Keyword Arguments:
        save_path {str} -- path to save figure to (default: {None})
    """

    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax = plt.axes()
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(columns)
    ax.set_xlabel("Feature importance (bigger = more important)")
    ax.set_title('Feature Importances')
    ax.barh(
        range(len(importances)), importances,
        alpha=0.75)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)


def plot_training_curves(losses, labels, save_path=None):
    """Plot training curves

    Arguments:
        losses {list of float} -- loss values
        labels {list of str} -- labels

    Keyword Arguments:
        save_path {str} -- path to save figure to (default: {None})
    """
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for loss, label in zip(losses, labels):
        ax.semilogy(loss, label=label)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print('Figure saved to: %s' % save_path)
