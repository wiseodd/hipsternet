import numpy as np


def cross_entropy(y_pred, y_train, model, lam):
    m = y_pred.shape[0]

    log_like = -np.log(y_pred[range(m), y_train])
    data_loss = np.sum(log_like) / m

    W1, W2, W3 = model['W1'], model['W2'], model['W3']

    reg_loss = 0.
    reg_loss += .5 * lam * np.sum(W1 * W1)
    reg_loss += .5 * lam * np.sum(W2 * W2)
    reg_loss += .5 * lam * np.sum(W3 * W3)

    loss = data_loss + reg_loss

    return loss
