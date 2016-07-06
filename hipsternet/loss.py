import numpy as np
import hipsternet.regularization as reg


def cross_entropy(y_pred, y_train, model, lam):
    m = y_pred.shape[0]
    W1, W2, W3 = model['W1'], model['W2'], model['W3']

    log_like = -np.log(y_pred[range(m), y_train])

    data_loss = np.sum(log_like) / m
    reg_loss = reg.l2_reg(W1, lam) + reg.l2_reg(W2, lam) + reg.l2_reg(W3, lam)

    loss = data_loss + reg_loss

    return loss
