import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.regularization as reg
import hipsternet.layer as l
import hipsternet.constant as c


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
        caches=dict(
            bn1_mean=np.zeros((1, H)),
            bn2_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )
    )

    return model


def train_step(model, X_train, y_train, lam=1e-3, p_dropout=.8, loss='cross_entropy'):
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
    y_pred, hiddens = forward(X_train, model, train=True, p_dropout=p_dropout)
    h1, h2, u1, u2, bn1_cache, bn2_cache = hiddens

    """
    Backprop
    """
    # Output layer
    if loss == 'cross_entropy':
        grad_y = loss_fun.dcross_entropy(y_pred, y_train)
    elif loss == 'hinge':
        grad_y = loss_fun.dhinge_loss(y_pred, y_train)

    # W3
    dW3 = h2.T @ grad_y
    dW3 += reg.dl2_reg(W3, lam)

    # b3
    db3 = np.sum(grad_y, axis=0)

    # h2
    dh2 = grad_y @ W3.T

    # ReLU
    dh2[h2 <= 0] = 0

    # Dropout h2
    dh2 *= u2

    # BatchNorm
    dh2, dgamma2, dbeta2 = l.batchnorm_backward(dh2, bn2_cache)

    # W2
    dW2 = h1.T @ dh2
    dW2 += reg.dl2_reg(W2, lam)

    # b2
    db2 = np.sum(dh2, axis=0)

    # h1
    dh1 = dh2 @ W2.T

    # ReLU
    dh1[h1 <= 0] = 0

    # Dropout h1
    dh1 *= u1

    # BatchNorm
    dh1, dgamma1, dbeta1 = l.batchnorm_backward(dh2, bn2_cache)

    # W1
    dW1 = X_train.T @ dh1
    dW1 += reg.dl2_reg(W1, lam)

    # b1
    db1 = np.sum(dh1, axis=0)

    model_grad = dict(
        W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
        gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
    )

    if loss == 'cross_entropy':
        cost = loss_fun.cross_entropy(y_pred, y_train, params, lam)
    elif loss == 'hinge':
        cost = loss_fun.hinge_loss(y_pred, y_train, params)

    return model_grad, cost


def forward(X, model, train=False, p_dropout=.5):
    m = X.shape[0]

    params = model['net_params']
    bn_params = model['caches']

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
        h1, bn1_cache, run_mean, run_var = l.batchnorm_forward(h1, gamma1, beta1, (bn1_mean, bn1_var))
        bn_params['bn1_mean'], bn_params['bn1_var'] = run_mean, run_var
    else:
        h1 = (h1 - bn1_mean) / np.sqrt(bn1_var + c.eps)
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
        h2, bn2_cache, run_mean, run_var = l.batchnorm_forward(h2, gamma2, beta2, (bn2_mean, bn2_var))
        bn_params['bn2_mean'], bn_params['bn2_var'] = run_mean, run_var
    else:
        h2 = (h2 - bn2_mean) / np.sqrt(bn2_var + c.eps)
        h2 = gamma2 * h2 + beta2

    # ReLU
    h2[h2 < 0] = 0

    if train:
        # Dropout
        u2 = np.random.binomial(1, p_dropout, size=h2.shape) / p_dropout
        h2 *= u2

    # Hidden to output
    score = h2 @ W3 + b3

    if train:
        return score, (h1, h2, u1, u2, bn1_cache, bn2_cache)
    else:
        return score


def predict_proba(X, model):
    prob = forward(X, model, False)
    return prob


def predict(X, model):
    return np.argmax(predict_proba(X, model), axis=1)
