import tensorflow as tf


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
