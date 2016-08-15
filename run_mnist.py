import numpy as np
import hipsternet.input_data as input_data
import hipsternet.neuralnet as nn
from hipsternet.solver import *
import sys


n_iter = 1000
alpha = 1e-3
mb_size = 64
n_experiment = 1
reg = 1e-5
print_after = 100
p_dropout = 0.8
loss = 'cross_ent'
nonlin = 'relu'
solver = 'adam'


def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean


if __name__ == '__main__':
    if len(sys.argv) > 1:
        net_type = sys.argv[1]
        valid_nets = ('ff', 'cnn')

        if net_type not in valid_nets:
            raise Exception('Valid network type are {}'.format(valid_nets))
    else:
        net_type = 'ff'

    mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)

    if net_type == 'cnn':
        img_shape = (1, 28, 28)
        X_train = X_train.reshape(-1, *img_shape)
        X_val = X_val.reshape(-1, *img_shape)
        X_test = X_test.reshape(-1, *img_shape)

    solvers = dict(
        sgd=sgd,
        momentum=momentum,
        nesterov=nesterov,
        adagrad=adagrad,
        rmsprop=rmsprop,
        adam=adam
    )

    solver_fun = solvers[solver]
    accs = np.zeros(n_experiment)

    print()
    print('Experimenting on {}'.format(solver))
    print()

    for k in range(n_experiment):
        print('Experiment-{}'.format(k + 1))

        # Reset model
        if net_type == 'ff':
            net = nn.FeedForwardNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin)
        elif net_type == 'cnn':
            net = nn.ConvNet(10, C, H=128)

        net = solver_fun(
            net, X_train, y_train, val_set=(X_val, y_val), mb_size=mb_size, alpha=alpha,
            n_iter=n_iter, print_after=print_after
        )

        y_pred = net.predict(X_test)
        accs[k] = np.mean(y_pred == y_test)

    print()
    print('Mean accuracy: {:.4f}, std: {:.4f}'.format(accs.mean(), accs.std()))
