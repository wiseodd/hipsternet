import sys
import numpy as np
import hipsternet.neuralnet as nn
import hipsternet.solver as solver


time_step = 10
n_iter = 1000000000
alpha = 1e-3
print_after = 1000

H = 64


if __name__ == '__main__':
    with open('data/text_data/japan.txt', 'r') as f:
        txt = f.read()

        X = []
        y = []

        char_to_idx = {char: i for i, char in enumerate(set(txt))}
        idx_to_char = {i: char for i, char in enumerate(set(txt))}

        X = np.array([char_to_idx[x] for x in txt])
        y = [char_to_idx[x] for x in txt[1:]]
        y.append(char_to_idx['.'])
        y = np.array(y)

    vocab_size = len(char_to_idx)

    if len(sys.argv) > 1:
        net_type = sys.argv[1]
        valid_nets = ('rnn', 'lstm', 'gru')

        if net_type not in valid_nets:
            raise Exception('Valid network type are {}'.format(valid_nets))
    else:
        net_type = 'lstm'

    if net_type == 'lstm':
        net = nn.LSTM(vocab_size, H=H, char2idx=char_to_idx, idx2char=idx_to_char)
    elif net_type == 'rnn':
        net = nn.RNN(vocab_size, H=H, char2idx=char_to_idx, idx2char=idx_to_char)
    elif net_type == 'gru':
        net = nn.GRU(vocab_size, H=H, char2idx=char_to_idx, idx2char=idx_to_char)

    solver.adam_rnn(
        net, X, y,
        alpha=alpha,
        mb_size=time_step,
        n_iter=n_iter,
        print_after=print_after
    )
