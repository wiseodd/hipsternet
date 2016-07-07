import numpy as np
import hipsternet.utils as util
import hipsternet.constant as c


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def batchnorm_forward(X, gamma, beta, cache, momentum=.9):
    running_mean, running_var = cache

    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + c.eps)
    out = gamma * X_norm + beta

    cache = (X, X_norm, mu, var, gamma, beta)

    running_mean = util.exp_running_avg(running_mean, mu, momentum)
    running_var = util.exp_running_avg(running_var, var, momentum)

    return out, cache, running_mean, running_var


def batchnorm_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + c.eps)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta
