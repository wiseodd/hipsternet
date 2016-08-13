import numpy as np
import hipsternet.utils as util
import hipsternet.constant as c
import copy
from sklearn.utils import shuffle as skshuffle


def get_minibatch(X, y, minibatch_size, shuffle=True):
    minibatches = []

    if shuffle:
        X, y = skshuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


def sgd(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            nn.model[layer] -= alpha * grad[layer]

    return nn


def momentum(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    velocity = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            nn.model[layer] -= velocity[layer]

    return nn


def nesterov(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    velocity = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        nn_ahead = copy.deepcopy(nn)
        nn_ahead.model.update({k: v + gamma * velocity[k] for k, v in nn.model.items()})
        grad, loss = nn_ahead.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            nn.model[layer] -= velocity[layer]

    return nn


def adagrad(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}

    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for k in grad:
            cache[k] += grad[k]**2
            nn.model[k] -= alpha * grad[k] / (np.sqrt(cache[k]) + c.eps)

    return nn


def rmsprop(nn, X_train, y_train, val_set=None, alpha=1e-3, mb_size=256, n_iter=2000, print_after=100):
    cache = {k: np.zeros_like(v) for k, v in nn.model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for k in grad:
            cache[k] = util.exp_running_avg(cache[k], grad[k]**2, gamma)
            nn.model[k] -= alpha * grad[k] / (np.sqrt(cache[k]) + c.eps)

    return nn


def adam(nn, X_train, y_train, val_set=None, alpha=0.001, mb_size=256, n_iter=2000, print_after=100):
    M = {k: np.zeros_like(v) for k, v in nn.model.items()}
    R = {k: np.zeros_like(v) for k, v in nn.model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, mb_size)

    if val_set:
        X_val, y_val = val_set

    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad, loss = nn.train_step(X_mini, y_mini)

        if iter % print_after == 0:
            if val_set:
                val_acc = util.accuracy(y_val, nn.predict(X_val))
                print('Iter-{} loss: {:.4f} validation: {:4f}'.format(iter, loss, val_acc))
            else:
                print('Iter-{} loss: {:.4f}'.format(iter, loss))

        for k in grad:
            M[k] = util.exp_running_avg(M[k], grad[k], beta1)
            R[k] = util.exp_running_avg(R[k], grad[k]**2, beta2)

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            nn.model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + c.eps)

    return nn


def adam_rnn(nn, X_train, y_train, alpha=0.001, mb_size=256, n_iter=2000, print_after=100):
    M = {k: np.zeros_like(v) for k, v in nn.model.items()}
    R = {k: np.zeros_like(v) for k, v in nn.model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, mb_size, shuffle=False)

    idx = 0
    state = nn.initial_state()
    smooth_loss = -np.log(1.0 / len(set(X_train)))

    for iter in range(1, n_iter + 1):
        t = iter

        if idx >= len(minibatches):
            idx = 0
            state = nn.initial_state()

        X_mini, y_mini = minibatches[idx]
        idx += 1

        if iter % print_after == 0:
            print("=========================================================================")
            print('Iter-{} loss: {:.4f}'.format(iter, smooth_loss))
            print("=========================================================================")

            sample = nn.sample(X_mini[0], state, 100)
            print(sample)

            print("=========================================================================")
            print()
            print()

        grad, loss, state = nn.train_step(X_mini, y_mini, state)
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        for k in grad:
            M[k] = util.exp_running_avg(M[k], grad[k], beta1)
            R[k] = util.exp_running_avg(R[k], grad[k]**2, beta2)

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            nn.model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + c.eps)

    return nn
