import numpy as np
import hipsternet.utils as util
import hipsternet.constant as c
import hipsternet.regularization as reg
from hipsternet.im2col import *


def fc_forward(X, W, b):
    out = X @ W + b
    cache = (W, X)
    return out, cache


def fc_backward(dout, cache):
    W, h = cache

    dW = h.T @ dout
    db = np.sum(dout, axis=0)
    dX = dout @ W.T

    return dX, dW, db


def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache <= 0] = 0
    return dX


def lrelu_forward(X, a=1e-3):
    out = np.maximum(a * X, X)
    cache = (X, a)
    return out, cache


def lrelu_backward(dout, cache):
    X, a = cache
    dX = dout.copy()
    dX[X < 0] *= a
    return dX


def sigmoid_forward(X):
    out = util.sigmoid(X)
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    return cache * (1. - cache) * dout


def tanh_forward(X):
    out = np.tanh(X)
    cache = out
    return out, cache


def tanh_backward(dout, cache):
    dX = (1 - cache**2) * dout
    return dX


def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache


def dropout_backward(dout, cache):
    dX = dout * cache
    return dX


def bn_forward(X, gamma, beta, cache, momentum=.9, train=True):
    running_mean, running_var = cache

    if train:
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + c.eps)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        running_mean = util.exp_running_avg(running_mean, mu, momentum)
        running_var = util.exp_running_avg(running_var, var, momentum)
    else:
        X_norm = (X - running_mean) / np.sqrt(running_var + c.eps)
        out = gamma * X_norm + beta
        cache = None

    return out, cache, running_mean, running_var


def bn_backward(dout, cache):
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


def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db


def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def maxpool_backward(dout, cache):
    def dmaxpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    return _pool_backward(dout, dmaxpool, cache)


def avgpool_forward(X, size=2, stride=2):
    def avgpool(X_col):
        out = np.mean(X_col, axis=0)
        cache = None
        return out, cache

    return _pool_forward(X, avgpool, size, stride)


def avgpool_backward(dout, cache):
    def davgpool(dX_col, dout_col, pool_cache):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    return _pool_backward(dout, davgpool, cache)


def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    out, pool_cache = pool_fun(X_col)

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX = dpool_fun(dX_col, dout_col, pool_cache)

    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX
