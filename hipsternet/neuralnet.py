import numpy as np
import hipsternet.loss as loss


def make_network(D, C, H=100):
    model = dict(
        net_params=dict(
            W1=np.random.randn(D, H) / np.sqrt(D / 2.),
            W2=np.random.randn(H, H) / np.sqrt(H / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((1, H)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C)),
            gamma1=np.ones((1, H)),
            gamma2=np.ones((1, H)),
            beta1=np.zeros((1, H)),
            beta2=np.zeros((1, H))
        ),
        etc_params=dict(
            bn1_mean=np.zeros((1, H)),
            bn2_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )
    )

    return model


def softmax(x):
    e_x = np.exp((x.T - np.max(x, axis=1)).T)
    return (e_x.T / e_x.sum(axis=1)).T


def batchnorm_forward(X, gamma, beta):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    out = gamma * X_norm + beta

    cache = (X, X_norm, mu, var, gamma, beta)

    return out, cache, mu, var


def batchnorm_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta


def train_step(model, X_train, y_train, lam=1e-3, p_dropout=.5):
    """
    Single training step over minibatch: forward, loss, backprop
    """
    params = model['net_params']

    m = X_train.shape[0]
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    b1, b2, b3 = params['b1'], params['b2'], params['b3']
    gamma1, gamma2 = params['gamma1'], params['gamma2']
    beta1, beta2 = params['beta1'], params['beta2']

    """
    Forward pass
    """
    prob, hiddens = _predict_proba(X_train, model, train=True, p_dropout=p_dropout)
    h1, h2, u1, u2, bn1_cache, bn2_cache = hiddens

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

    # BatchNorm
    dh2, dgamma2, dbeta2 = batchnorm_backward(dh2, bn2_cache)

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

    # BatchNorm
    dh1, dgamma1, dbeta1 = batchnorm_backward(dh2, bn2_cache)

    # W1
    dW1 = X_train.T @ dh1
    dW1 += lam * W1

    # b1
    db1 = np.sum(dh1, axis=0)

    model_grad = dict(
        W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
        gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
    )

    cost = loss.cross_entropy(prob, y_train, params, lam)

    return model_grad, cost


def _predict_proba(X, model, train=False, p_dropout=.5):
    m = X.shape[0]

    params = model['net_params']
    bn_params = model['etc_params']

    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    b1, b2, b3 = params['b1'], params['b2'], params['b3']
    gamma1, gamma2 = params['gamma1'], params['gamma2']
    beta1, beta2 = params['beta1'], params['beta2']
    bn1_mean, bn2_mean = bn_params['bn1_mean'], bn_params['bn2_mean']
    bn1_var, bn2_var = bn_params['bn1_var'], bn_params['bn2_var']

    u1, u2 = None, None
    bn1_cache, bn2_cache = None, None

    # Input to hidden
    h1 = X @ W1 + b1

    # BatchNorm
    if train:
        h1, bn1_cache, mu, var = batchnorm_forward(h1, gamma1, beta1)
        bn_params['bn1_mean'] = .9 * bn_params['bn1_mean'] + .1 * mu
        bn_params['bn1_var'] = .9 * bn_params['bn1_var'] + .1 * var
    else:
        h1 = (h1 - bn_params['bn1_mean']) / np.sqrt(bn_params['bn1_var'] + 1e-8)
        h1 = gamma1 * h1 + beta1

    # ReLU
    h1[h1 < 0] = 0

    if train:
        # Dropout
        u1 = np.random.binomial(1, p_dropout, size=h1.shape) / p_dropout
        h1 *= u1

    # Hidden to hidden
    h2 = h1 @ W2 + b2

    # BatchNorm
    if train:
        h2, bn2_cache, mu, var = batchnorm_forward(h2, gamma2, beta2)
        bn_params['bn2_mean'] = .9 * bn_params['bn2_mean'] + .1 * mu
        bn_params['bn2_var'] = .9 * bn_params['bn2_var'] + .1 * var
    else:
        h2 = (h2 - bn_params['bn2_mean']) / np.sqrt(bn_params['bn2_var'] + 1e-8)
        h2 = gamma2 * h2 + beta2

    # ReLU
    h2[h2 < 0] = 0

    if train:
        # Dropout
        u2 = np.random.binomial(1, p_dropout, size=h2.shape) / p_dropout
        h2 *= u2

    # Hidden to output
    score = h2 @ W3 + b3
    prob = softmax(score)

    if train:
        return prob, (h1, h2, u1, u2, bn1_cache, bn2_cache)
    else:
        return prob


def predict_proba(X, model):
    prob = _predict_proba(X, model, False)
    return prob


def predict(X, model):
    return np.argmax(predict_proba(X, model), axis=1)
