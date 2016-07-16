import numpy as np
import hipsternet.utils as util
import hipsternet.constant as c
import hipsternet.regularization as reg
from hipsternet.im2col import *


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def fc_forward(X, W, b):
    return X @ W + b


def fc_backward(dinput, h, W, input_layer=False, lam=1e-3):
    dW = h.T @ dinput
    dW += reg.dl2_reg(W, lam)
    db = np.sum(dinput, axis=0)

    dh = None

    if not input_layer:
        dh = dinput @ W.T

    return dh, dW, db


def relu_forward(h):
    return np.maximum(h, 0)


def relu_backward(dh, h):
    out = dh.copy()
    out[h <= 0] = 0
    return out


def lrelu_forward(h, a=1e-3):
    return np.maximum(a * h, h)


def lrelu_backward(dh, h, a=1e-3):
    out = dh.copy()
    out[h < 0] *= a
    return out


def sigmoid_forward(h):
    return 1. / np.log(1 + np.exp(-h))


def sigmoid_backward(dh, cache=None):
    return sigmoid_forward(dh) * (1 - sigmoid_forward(dh))


def tanh_forward(h):
    return np.tanh(h)


def tanh_backward(dh, cache=None):
    return 1 - np.tanh(dh)**2


def dropout_forward(h, p_dropout):
    u = np.random.binomial(1, p_dropout, size=h.shape) / p_dropout
    return h * u, u


def dropout_backward(dh, dropout_mask):
    return dh * dropout_mask


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
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
    max_idx = np.argmax(X_col, axis=0)

    out = X_col[max_idx, range(max_idx.size)]
    out = out.reshape(d, h_out, w_out, n)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, size, stride, X_col, max_idx)

    return out, cache


def maxpool_backward(dout, cache):
    X, size, stride, X_col, max_idx = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_flat = dout.transpose(2, 3, 0, 1).ravel()

    dX_col[max_idx, range(max_idx.size)] = dout_flat
    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX
