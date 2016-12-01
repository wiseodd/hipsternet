import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def feedforward_net(D, H, C):
    X = tf.placeholder(tf.float32, shape=[None, D])
    y = tf.placeholder(tf.float32, shape=[None, C])

    Wxh = tf.Variable(xavier_init([D, H]))
    bxh = tf.Variable(tf.zeros(shape=[H]))

    Whh = tf.Variable(xavier_init([H, H]))
    bhh = tf.Variable(tf.zeros(shape=[H]))

    Why = tf.Variable(xavier_init([H, C]))
    bhy = tf.Variable(tf.zeros(shape=[C]))

    h1 = tf.nn.relu(tf.matmul(X, Wxh) + bxh)
    h2 = tf.nn.relu(tf.matmul(h1, Whh) + bhh)
    prob = tf.nn.softmax(tf.matmul(h2, Why) + bhy)

    loss = -tf.reduce_mean(y * tf.log(prob))

    return X, y, prob, loss


def convnet(D, H, C):
    X = tf.placeholder(tf.float32, shape=[None, *D])
    y = tf.placeholder(tf.float32, shape=[None, C])

    Wconv1 = tf.Variable(xavier_init([3, 3, 1, 10]))
    bconv1 = tf.Variable(tf.zeros(shape=[10]))

    Wfc1 = tf.Variable(xavier_init([14 * 14 * 10, H]))
    bfc1 = tf.Variable(tf.zeros(shape=[H]))

    Wfc2 = tf.Variable(xavier_init([H, C]))
    bfc2 = tf.Variable(tf.zeros(shape=[C]))

    hconv1 = tf.nn.relu(tf.nn.conv2d(X, Wconv1, [1, 1, 1, 1], padding='SAME') + bconv1)
    hpool1 = tf.nn.max_pool(hconv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hpool1 = tf.reshape(hpool1, shape=[-1, 14 * 14 * 10])
    h = tf.nn.relu(tf.matmul(hpool1, Wfc1) + bfc1)
    prob = tf.nn.softmax(tf.matmul(h, Wfc2) + bfc2)

    loss = -tf.reduce_mean(y * tf.log(prob))

    return X, y, prob, loss


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


if __name__ == '__main__':
    alpha = 1e-3

    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    if len(sys.argv) > 1:
        net_type = sys.argv[1]
        valid_nets = ('ff', 'cnn')

        if net_type not in valid_nets:
            raise Exception('Valid network type are {}'.format(valid_nets))
    else:
        net_type = 'ff'

    D, C = X_train.shape[1], y_train.shape[1]
    H = 64
    M = 128

    if net_type == 'cnn':
        D = [28, 28, 1]
        X, y, forward_step, loss = convnet(D, H, C)
        X_val = X_val.reshape([-1, 28, 28, 1])
    elif net_type == 'ff':
        X, y, forward_step, loss = feedforward_net(D, H, C)

    solver = tf.train.RMSPropOptimizer(alpha)
    train_step = solver.minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(5000):
        X_mb, y_mb = mnist.train.next_batch(M)

        if net_type == 'cnn':
            X_mb = X_mb.reshape([-1, 28, 28, 1])

        _, loss_val = sess.run([train_step, loss], feed_dict={X: X_mb, y: y_mb})

        if i % 100 == 0:
            y_pred = sess.run(forward_step, feed_dict={X: X_val})
            acc = accuracy(y_val, y_pred)

            print('Iter: {} Loss: {:.4f} Validation: {}'.format(i, loss_val, acc))
