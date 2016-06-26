import numpy as np
import hipsternet.loss as loss


def make_network(D, C, H=100):
    model = dict(
        W1=np.random.randn(D, H) / np.sqrt(D / 2.),
        W2=np.random.randn(H, H) / np.sqrt(H / 2.),
        W3=np.random.randn(H, C) / np.sqrt(H / 2.),
        b1=np.zeros((1, H)),
        b2=np.zeros((1, H)),
        b3=np.zeros((1, C))
    )

    return model


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def train_step(model, X_train, y_train, lam=1e-3, p_dropout=.5):
    """
    Single training step over minibatch: forward, loss, backprop
    """

    m = X_train.shape[0]
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    b1, b2, b3 = model['b1'], model['b2'], model['b3']

    """
    Forward pass
    """

    # Input to hidden
    h1 = X_train @ W1 + b1
    h1[h1 < 0] = 0

    # Dropout
    u1 = np.random.binomial(1, p_dropout, size=h1.shape) / p_dropout
    h1 *= u1

    # Hidden to hidden
    h2 = h1 @ W2 + b2
    h2[h2 < 0] = 0

    # Dropout
    u2 = np.random.binomial(1, p_dropout, size=h2.shape) / p_dropout
    h2 *= u2

    # Hidden to output
    score = h2 @ W3 + b3
    prob = softmax(score)

    """
    Backprop
    """

    # Softmax layer
    grad_y = prob.copy()
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    # W3
    dW3 = h2.T @ grad_y
    dW3 += lam * W3

    # b3
    db3 = np.sum(grad_y, axis=0)

    # h2
    dh2 = grad_y @ W3.T

    # ReLU
    dh2[h2 <= 0] = 0

    # Dropout h2
    dh2 *= u2

    # W2
    dW2 = h1.T @ dh2
    dW2 += lam * W2

    # b2
    db2 = np.sum(dh2, axis=0)

    # h1
    dh1 = dh2 @ W2.T

    # ReLU
    dh1[h1 <= 0] = 0

    # Dropout h1
    dh1 *= u1

    # W1
    dW1 = X_train.T @ dh1
    dW1 += lam * W1

    # b1
    db1 = np.sum(dh1, axis=0)

    model_grad = dict(W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3)
    cost = loss.cross_entropy(prob, y_train, model, lam)

    return model_grad, cost


def predict_proba(X, model):
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    b1, b2, b3 = model['b1'], model['b2'], model['b3']

    # Input to hidden
    h1 = X @ W1 + b1
    h1[h1 < 0] = 0

    # Hidden to hidden
    h2 = h1 @ W2 + b2
    h2[h2 < 0] = 0

    # Hidden to output
    score = h2 @ W3 + b3
    prob = softmax(score)

    return prob


def predict(X, model):
    return np.argmax(predict_proba(X, model), axis=1)
