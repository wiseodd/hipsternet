import numpy as np
import hipsternet.neuralnet as nn
import hipsternet.solver as solver


time_step = 10


if __name__ == '__main__':
    with open('text_data/japan.txt', 'r') as f:
        txt = f.read()

        X = []
        y = []

        char_to_idx = {char: i for i, char in enumerate(set(txt))}

        X = np.array([char_to_idx[x] for x in txt])
        y = [char_to_idx[x] for x in txt[1:]]
        y.append(char_to_idx['.'])
        y = np.array(y)

    vocab_size = len(char_to_idx)

    net = nn.LSTM(vocab_size, vocab_size, H=vocab_size)
    solver.adam(net, X, y, mb_size=time_step, n_iter=5000, print_after=100)
