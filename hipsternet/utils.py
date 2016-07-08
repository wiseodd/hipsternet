import numpy as np


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)
