import numpy as np
import hipsternet.neuralnet as nn
import hipsternet.solver as solver


batch_size = 10


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

    net = nn.RNN(vocab_size, vocab_size, H=64)
    solver.adam(net, X, y, mb_size=10, n_iter=10000)
