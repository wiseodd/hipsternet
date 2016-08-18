import tensorflow as tf
import numpy as np
import models
from tensorflow.examples.tutorials.mnist import input_data


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


if __name__ == '__main__':
    alpha = 1e-3

    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    D, C = X_train.shape[1], y_train.shape[1]
    H = 64
    M = 128

    X, y, forward_step, loss = models.feedforward_net(D, H, C)

    solver = tf.train.RMSPropOptimizer(alpha)
    train_step = solver.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(5000):
        X_mb, y_mb = mnist.train.next_batch(M)
        _, loss_val = sess.run([train_step, loss], feed_dict={X: X_mb, y: y_mb})

        if i % 100 == 0:
            y_pred = sess.run(forward_step, feed_dict={X: X_val})
            acc = accuracy(y_val, y_pred)

            print('Iter: {} Loss: {:.4f} Validation: {}'.format(i, loss_val, acc))
