import numpy as np
import hipsternet.input_data as input_data
import hipsternet.solver as solver
from hipsternet.neuralnet import NeuralNet


n_iter = 2000
alpha = 1e-3
mb_size = 100
n_experiment = 1
reg = 1e-3
print_after = 100
p_dropout = 0.8
loss = 'cross_ent'


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    # Normalization
    X_mean = X_train.mean()

    X_train = X_train - X_mean
    X_test = X_test - X_mean

    solvers = dict(
        sgd=solver.sgd,
        momentum=solver.momentum,
        nesterov=solver.nesterov,
        adagrad=solver.adagrad,
        rmsprop=solver.rmsprop,
        adam=solver.adam
    )

    solver_accs = {k: np.zeros(n_experiment) for k in solvers}

    for solver_name, solver_fun in solvers.items():
        print('Experimenting on {}'.format(solver_name))

        for k in range(n_experiment):
            print('Experiment-{}'.format(k))

            # Reset model
            nn = NeuralNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss)

            nn = solver_fun(
                nn, X_train, y_train, mb_size=mb_size, alpha=alpha, n_iter=n_iter, print_after=print_after
            )

            y_pred = nn.predict(X_test)

            solver_accs[solver_name][k] = np.mean(y_pred == y_test)

    print()

    for k, v in solver_accs.items():
        print('{} => mean accuracy: {:.4f}, std: {:.4f}'.format(k, v.mean(), v.std()))
