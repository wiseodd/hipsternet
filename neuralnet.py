import numpy as np
import input_data


def make_network(D, C, H=100):
    W1_size = D * H
    W2_size = H * C

    model = dict(
        W1=np.random.randn(D, H) / np.sqrt(D / 2.),
        W2=np.random.randn(H, C) / np.sqrt(H / 2.),
        b1=np.zeros((1, H)),
        b2=np.zeros((1, C))
    )

    return model


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def train_step(model, X_train, y_train, lam=1e-3, p_dropout=.5):
    """
    Single training step over minibatch: forward-loss-backprop
    """

    m = X_train.shape[0]
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']

    """
    Forward pass
    """

    # Input to hidden
    h = X_train @ W1 + b1
    h[h < 0] = 0

    # Dropout
    dropout_mask = np.random.rand(*h.shape) < p_dropout
    h = h * dropout_mask / p_dropout

    # Hidden to output
    score = h @ W2 + b2
    prob = softmax(score)

    """
    Compute loss
    """

    log_like = -np.log(prob[range(m), y_train])
    data_loss = np.sum(log_like) / m

    reg_loss = 0.
    reg_loss += .5 * lam * np.sum(W1 * W1)
    reg_loss += .5 * lam * np.sum(W2 * W2)

    loss = data_loss + reg_loss

    """
    Backprop
    """

    # Softmax layer
    grad_y = prob.copy()
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    # W2
    dW2 = h.T @ grad_y
    dW2 += lam * W2

    # b2
    db2 = np.sum(grad_y, axis=0)

    # h
    dh = grad_y @ W2.T

    # Dropout
    dh = dh * dropout_mask / p_dropout

    # ReLU
    dh[h <= 0] = 0

    # W1
    dW1 = X_train.T @ dh
    dW1 += lam * W1

    # b1
    db1 = np.sum(dh, axis=0)

    model_grad = dict(W1=dW1, W2=dW2, b1=db1, b2=db2)

    return model_grad, loss


def predict_proba(X, model):
    # Input to hidden
    h = X @ model['W1'] + model['b1']
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'] + model['b2'])

    return prob


def predict(X, model):
    return np.argmax(predict_proba(X, model), axis=1)
