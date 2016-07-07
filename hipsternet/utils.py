import numpy as np


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new
