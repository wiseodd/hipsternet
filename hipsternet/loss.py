import numpy as np
import hipsternet.regularization as reg


def cross_entropy(y_pred, y_train, model, lam=1e-3):
    m = y_pred.shape[0]

    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train])

    data_loss = np.sum(log_like) / m

    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    reg_loss = reg.l2_reg(W1, lam) + reg.l2_reg(W2, lam) + reg.l2_reg(W3, lam)

    loss = data_loss + reg_loss

    return loss


def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = softmax(y_pred)
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    return grad_y


def hinge_loss(y_pred, y_train, model, lam=1e-3, delta=1):
    m = y_pred.shape[0]

    margins = (y_pred.T - y_pred[range(m), y_train]).T + delta
    margins[margins < 0] = 0
    margins[range(m), y_train] = 0

    data_loss = np.sum(margins) / m

    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    reg_loss = reg.l2_reg(W1, lam) + reg.l2_reg(W2, lam) + reg.l2_reg(W3, lam)

    loss = data_loss + reg_loss

    return loss


def dhinge_loss(y_pred, y_train, margin=1):
    m = y_pred.shape[0]

    margins = (y_pred.T - y_pred[range(m), y_train]).T + 1.
    margins[range(m), y_train] = 0

    grad_y = (margins > 0).astype(float)
    grad_y[range(m), y_train] = -np.sum(grad_y, axis=1)
    grad_y /= m

    return grad_y


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T
