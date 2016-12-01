import tensorflow as tf
import numpy as np


def onehot(vocab_size, idx):
    x = np.zeros(shape=vocab_size)
    x[idx] = 1.
    return x


if __name__ == '__main__':
    with open('../data/text_data/japan.txt', 'r') as f:
        txt = f.read()

        X = []
        y = []

        char_to_idx = {char: i for i, char in enumerate(set(txt))}
        idx_to_char = {i: char for i, char in enumerate(set(txt))}

        vocab_size = len(set(txt))

        X = np.array([onehot(vocab_size, char_to_idx[x]) for x in txt])
        X = X.reshape(1, *X.shape)

        y = X[:, 1:, :]
        y_last = onehot(vocab_size, char_to_idx['.'])
        y = np.column_stack((y, y_last.reshape(1, 1, 71)))

    batch_size = 1
    step_size = 25
    H = 64

    words = tf.placeholder(dtype=tf.int32, shape=[batch_size, step_size])
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size, step_size])

    lstm = tf.nn.rnn_cell.BasicLSTMCell(H, state_is_tuple=True)
    Why = tf.Variable(tf.random_normal(shape=[H, vocab_size], stddev=.001))
    bhy = tf.Variable(tf.zeros(vocab_size))

    initial_state = state = tf.zeros(shape=[batch_size, lstm.state_size])
    loss = 0.

    for i in range(step_size):
        h, state = lstm(words[:, i], state)
        prob = tf.nn.softmax(tf.matmul(h, Why) + bhy)
        loss += -tf.reduce_mean(y[:, i] * tf.log(prob))

    final_state = state

    sess = tf.Session()
    total_loss = 0.

    for i in range(10000):
        if i == 0 or idx >= X.shape[0]:
            idx = 0
            np_state = sess.run(initial_state)

        np_state, seq_loss = sess.run(
            [final_state, loss],
            feed_dict={words: X[:, idx:idx + step_size, :], initial_state: np_state}
        )
        total_loss += seq_loss
