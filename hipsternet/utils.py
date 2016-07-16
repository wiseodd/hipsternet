import numpy as np


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def onehot(labels):
    y = np.zeros([labels.size, np.max(labels) + 1])
    y[range(labels.size), labels] = 1.
    return y


def conv_2d(X, kernel, stride=1, padding=1):
    if not is_square(X) or not is_square(kernel):
        raise Exception('Image and kernel must be a square matrix!')

    X_pad = zeropad_image(X, padding)

    m = X.shape[0]
    w = X_pad.shape[0]
    k = kernel.shape[0]

    out_dim = (m - k + 2 * padding) / stride + 1

    if not out_dim.is_integer():
        raise Exception('Convolution parameters invalid! Please check the input, kernel, stride, and padding size!')

    out_dim = int(out_dim)
    out = np.zeros(shape=[out_dim, out_dim])

    for i, ii in enumerate(range(0, w - k + 1, stride)):
        for j, jj in enumerate(range(0, w - k + 1, stride)):
            out[i, j] = np.sum(X_pad[ii:ii + k, jj:jj + k] * kernel)

    return out


def maxpool_2d(X, k=2, stride=2):
    if not is_square(X):
        raise Exception('Image must be a square matrix!')

    m = X.shape[0]
    out_dim = (m - k) / stride + 1

    if not out_dim.is_integer():
        raise Exception('Pooling parameters invalid! Please check the input, pool size, and stride!')

    out_dim = int(out_dim)
    out = np.zeros(shape=[out_dim, out_dim])

    cache = []

    for i, ii in enumerate(range(0, m - k + 1, stride)):
        idxs = []

        for j, jj in enumerate(range(0, m - k + 1, stride)):
            patch = X[ii:ii + k, jj:jj + k]
            x, y = np.unravel_index(np.argmax(patch), patch.shape)
            out[i, j] = patch[x, y]
            idxs.append((ii + x, jj + y))

        cache.append(idxs)

    return out, cache


def zeropad_image(X, pad=1):
    m, n = X.shape

    X_pad = np.zeros(shape=[m + 2 * pad, n + 2 * pad])
    X_pad[pad:m + pad, pad:n + pad] = X

    return X_pad


def is_square(X):
    return X.shape[0] == X.shape[1]
