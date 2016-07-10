import numpy as np
import hipsternet.input_data as input_data
from hipsternet.solver import *
from hipsternet.neuralnet import NeuralNet


n_iter = 5000
alpha = 1e-3
mb_size = 100
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
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)

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
        nn = NeuralNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin)

        nn = solver_fun(
            nn, X_train, y_train, val_set=(X_val, y_val), mb_size=mb_size, alpha=alpha,
            n_iter=n_iter, print_after=print_after
        )

        y_pred = nn.predict(X_test)
        accs[k] = np.mean(y_pred == y_test)

    print()
    print('Mean accuracy: {:.4f}, std: {:.4f}'.format(accs.mean(), accs.std()))
