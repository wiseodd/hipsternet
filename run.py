import numpy as np
import input_data
import neuralnet as nn
import optimization as optim


n_iter = 2000
alpha = 1e-3
mb_size = 256
n_experiment = 1
reg = 1e-3
print_after = 100


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    # Normalization
    X_mean = X_train.mean()

    X_train = X_train - X_mean
    X_test = X_test - X_mean

    algos = dict(
        sgd=optim.sgd,
        momentum=optim.momentum,
        nesterov=optim.nesterov,
        adagrad=optim.adagrad,
        rmsprop=optim.rmsprop,
        adam=optim.adam
    )

    algo_accs = {k: np.zeros(n_experiment) for k in algos}

    for algo_name, algo in algos.items():
        print('Experimenting on {}'.format(algo_name))

        for k in range(n_experiment):
            print('Experiment-{}'.format(k))

            # Reset model
            model = nn.make_network(D, C, H=1024)

            model = algo(
                model, X_train, y_train, mb_size=mb_size, alpha=alpha, n_iter=n_iter, print_after=print_after
            )

            y_pred = nn.predict(X_test, model)

            algo_accs[algo_name][k] = np.mean(y_pred == y_test)

    print()

    for k, v in algo_accs.items():
        print('{} => mean accuracy: {:.4f}, std: {:.4f}'.format(k, v.mean(), v.std()))
