import tensorflow as tf


def xavier_init(in_dim, out_dim):
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=[in_dim, out_dim], stddev=xavier_stddev)


def feedforward_net(D, H, C):
    X = tf.placeholder(tf.float32, shape=[None, D])
    y = tf.placeholder(tf.float32, shape=[None, C])

    Wxh = tf.Variable(xavier_init(D, H))
    bxh = tf.Variable(tf.zeros(shape=[H]))

    Whh = tf.Variable(xavier_init(H, H))
    bhh = tf.Variable(tf.zeros(shape=[H]))

    Why = tf.Variable(xavier_init(H, C))
    bhy = tf.Variable(tf.zeros(shape=[C]))

    h1 = tf.nn.relu(tf.matmul(X, Wxh) + bxh)
    h2 = tf.nn.relu(tf.matmul(h1, Whh) + bhh)
    prob = tf.nn.softmax(tf.matmul(h2, Why) + bhy)

    loss = -tf.reduce_mean(y * tf.log(prob))

    return X, y, prob, loss
