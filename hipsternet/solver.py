import numpy as np
import hipsternet.utils as util
import hipsternet.constant as c
import copy


def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)

    return Z[:, :-1], Z[:, -1].astype(int)


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


def sgd(nn, X_train, y_train, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        for layer in grad:
            nn.model[layer] -= alpha * grad[layer]

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn


def momentum(nn, X_train, y_train, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    velocity = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            nn.model[layer] -= velocity[layer]

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn


def nesterov(nn, X_train, y_train, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    velocity = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        nn_ahead = copy.deepcopy(nn)
        nn_ahead.model.update({k: v + gamma * velocity[k] for k, v in nn.model.items()})
        grad, loss = nn_ahead.train_step(X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            nn.model[layer] -= velocity[layer]

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn


def adagrad(nn, X_train, y_train, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}

    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        for k in grad:
            cache[k] += grad[k]**2
            nn.model[k] -= alpha * grad[k] / (np.sqrt(cache[k]) + c.eps)

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn


def rmsprop(nn, X_train, y_train, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        for k in grad:
            cache[k] = util.exp_running_avg(cache[k], grad[k]**2, gamma)
            nn.model[k] -= alpha * grad[k] / (np.sqrt(cache[k]) + c.eps)

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn


def adam(nn, X_train, y_train, alpha=0.001, mb_size=256, n_iter=2000, print_after=100):
    M = {k: np.zeros_like(v) for k, v in nn.model.items()}
    R = {k: np.zeros_like(v) for k, v in nn.model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, mb_size)

    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        for k in grad:
            M[k] = util.exp_running_avg(M[k], grad[k], beta1)
            R[k] = util.exp_running_avg(R[k], grad[k]**2, beta2)

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            nn.model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + c.eps)

        if iter % print_after == 0:
            print('Iter-{} loss: {}'.format(iter, loss))

    return nn
