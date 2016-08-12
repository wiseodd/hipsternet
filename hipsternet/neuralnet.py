import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.layer as l
import hipsternet.regularization as reg
import hipsternet.utils as util


class NeuralNet(object):

    loss_funs = dict(
        cross_ent=loss_fun.cross_entropy,
        hinge=loss_fun.hinge_loss,
        squared=loss_fun.squared_loss,
        l2_regression=loss_fun.l2_regression,
        l1_regression=loss_fun.l1_regression
    )

    dloss_funs = dict(
        cross_ent=loss_fun.dcross_entropy,
        hinge=loss_fun.dhinge_loss,
        squared=loss_fun.dsquared_loss,
        l2_regression=loss_fun.dl2_regression,
        l1_regression=loss_fun.dl1_regression
    )

    forward_nonlins = dict(
        relu=l.relu_forward,
        lrelu=l.lrelu_forward,
        sigmoid=l.sigmoid_forward,
        tanh=l.tanh_forward
    )

    backward_nonlins = dict(
        relu=l.relu_backward,
        lrelu=l.lrelu_backward,
        sigmoid=l.sigmoid_backward,
        tanh=l.tanh_backward
    )

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        if loss not in NeuralNet.loss_funs.keys():
            raise Exception('Loss function must be in {}!'.format(NeuralNet.loss_funs.keys()))

        if nonlin not in NeuralNet.forward_nonlins.keys():
            raise Exception('Nonlinearity must be in {}!'.format(NeuralNet.forward_nonlins.keys()))

        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss
        self.forward_nonlin = NeuralNet.forward_nonlins[nonlin]
        self.backward_nonlin = NeuralNet.backward_nonlins[nonlin]
        self.mode = 'classification'

        if 'regression' in loss:
            self.mode = 'regression'

    def train_step(self, X_train, y_train):
        """
        Single training step over minibatch: forward, loss, backprop
        """
        y_pred, cache = self.forward(X_train, train=True)
        loss = self.loss_funs[self.loss](self.model, y_pred, y_train, self.lam)
        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def predict_proba(self, X):
        score, _ = self.forward(X, False)
        return util.softmax(score)

    def predict(self, X):
        if self.mode == 'classification':
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            score, _ = self.forward(X, False)
            y_pred = np.round(score)
            return y_pred

    def forward(self, X, train=False):
        raise NotImplementedError()

    def backward(self, y_pred, y_train, cache):
        raise NotImplementedError()

    def _init_model(self, D, C, H):
        raise NotImplementedError()


class FeedForwardNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        gamma1, gamma2 = self.model['gamma1'], self.model['gamma2']
        beta1, beta2 = self.model['beta1'], self.model['beta2']

        u1, u2 = None, None
        bn1_cache, bn2_cache = None, None

        # First layer
        h1, h1_cache = l.fc_forward(X, self.model['W1'], self.model['b1'])
        bn1_cache = (self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'])
        h1, bn1_cache, run_mean, run_var = l.bn_forward(h1, gamma1, beta1, bn1_cache, train=train)
        h1, nl_cache1 = self.forward_nonlin(h1)

        self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'] = run_mean, run_var

        if train:
            h1, u1 = l.dropout_forward(h1, self.p_dropout)

        # Second layer
        h2, h2_cache = l.fc_forward(h1, self.model['W2'], self.model['b2'])
        bn2_cache = (self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'])
        h2, bn2_cache, run_mean, run_var = l.bn_forward(h2, gamma2, beta2, bn2_cache, train=train)
        h2, nl_cache2 = self.forward_nonlin(h2)

        self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'] = run_mean, run_var

        if train:
            h2, u2 = l.dropout_forward(h2, self.p_dropout)

        # Third layer
        score, score_cache = l.fc_forward(h2, self.model['W3'], self.model['b3'])

        cache = (X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache)

        return score, cache

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # Third layer
        dh2, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dW3 += reg.dl2_reg(self.model['W3'], self.lam)
        dh2 = self.backward_nonlin(dh2, nl_cache2)
        dh2 = l.dropout_backward(dh2, u2)
        dh2, dgamma2, dbeta2 = l.bn_backward(dh2, bn2_cache)

        # Second layer
        dh1, dW2, db2 = l.fc_backward(dh2, h2_cache)
        dW2 += reg.dl2_reg(self.model['W2'], self.lam)
        dh1 = self.backward_nonlin(dh1, nl_cache1)
        dh1 = l.dropout_backward(dh1, u1)
        dh1, dgamma1, dbeta1 = l.bn_backward(dh1, bn1_cache)

        # First layer
        _, dW1, db1 = l.fc_backward(dh1, h1_cache)
        dW1 += reg.dl2_reg(self.model['W1'], self.lam)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
            gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, H) / np.sqrt(D / 2.),
            W2=np.random.randn(H, H) / np.sqrt(H / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((1, H)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C)),
            gamma1=np.ones((1, H)),
            gamma2=np.ones((1, H)),
            beta1=np.zeros((1, H)),
            beta2=np.zeros((1, H))
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((1, H)),
            bn2_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )


class ConvNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # Conv-1
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = l.relu_forward(h1)

        # Pool-1
        hpool, hpool_cache = l.maxpool_forward(h1)
        h2 = hpool.ravel().reshape(X.shape[0], -1)

        # FC-7
        h3, h3_cache = l.fc_forward(h2, self.model['W2'], self.model['b2'])
        h3, nl_cache3 = l.relu_forward(h3)

        # Softmax
        score, score_cache = l.fc_forward(h3, self.model['W3'], self.model['b3'])

        return score, (X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3)

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3 = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # FC-7
        dh3, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dh3 = self.backward_nonlin(dh3, nl_cache3)

        dh2, dW2, db2 = l.fc_backward(dh3, h3_cache)
        dh2 = dh2.ravel().reshape(hpool.shape)

        # Pool-1
        dpool = l.maxpool_backward(dh2, hpool_cache)

        # Conv-1
        dh1 = self.backward_nonlin(dpool, nl_cache1)
        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(D * 14 * 14, H) / np.sqrt(D * 14 * 14 / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((D, 1)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C))
        )


class RNN(NeuralNet):

    def __init__(self, D, C, H, char2idx, idx2char, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        self.D = D
        self.H = H
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, h, train=True):
        Wxh, Whh, Why = self.model['Wxh'], self.model['Whh'], self.model['Why']
        bh, by = self.model['bh'], self.model['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        hprev = h.copy()

        h, h_cache = l.tanh_forward(X_one_hot @ Wxh + hprev @ Whh + bh)
        y, y_cache = l.fc_forward(h, Why, by)

        cache = (X_one_hot, Whh, h, hprev, y, h_cache, y_cache)

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        X, Whh, h, hprev, y, h_cache, y_cache = cache

        # Softmax gradient
        dy = loss_fun.dcross_entropy(y_pred, y_train)

        # Hidden to output gradient
        dh, dWhy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next
        dby = dby.reshape((1, -1))

        # tanh
        dh = l.tanh_backward(dh, h_cache)

        # Hidden gradient
        dbh = dh
        dWhh = hprev.T @ dh
        dWxh = X.T @ dh
        dh_next = dh @ Whh.T

        grad = dict(Wxh=dWxh, Whh=dWhh, Why=dWhy, bh=dbh, by=dby)

        return grad, dh_next

    def train_step(self, X_train, y_train, h=None):
        ys = []
        caches = []
        loss = 0.

        if h is None:
            h = np.zeros((1, self.H))

        # Forward
        for x, y in zip(X_train, y_train):
            y_pred, h, cache = self.forward(x, h, train=True)
            loss += loss_fun.cross_entropy(self.model, y_pred, y, lam=0)
            ys.append(y_pred)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for t in reversed(range(len(X_train))):
            grad, dh_next = self.backward(ys[t], y_train[t], dh_next, caches[t])

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, h

    def sample(self, X_seed, h, size=100):
        chars = [self.idx2char[X_seed]]
        idx_list = list(range(self.vocab_size))
        X = X_seed

        for _ in range(size - 1):
            prob, h, _ = self.forward(X, h, train=False)
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.idx2char[idx])
            X = idx

        return ''.join(chars)

    def _init_model(self, D, C, H):
        self.model = dict(
            Wxh=np.random.randn(D, H) / np.sqrt(D / 2.),
            Whh=np.random.randn(H, H) / np.sqrt(H / 2.),
            Why=np.random.randn(H, D) / np.sqrt(C / 2.),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )


class LSTM(RNN):

    def __init__(self, D, C, H, char2idx, idx2char, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        self.D = D
        self.H = H
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        super().__init__(D, C, H, char2idx, idx2char, lam, p_dropout, loss, nonlin)

    def forward(self, X, h_old, c_old, train=True):
        m = self.model
        Wf, Wi, Wc, Wo = m['Wf'], m['Wi'], m['Wc'], m['Wo']
        bf, bi, bc, bo = m['bf'], m['bi'], m['bc'], m['bo']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        X = np.column_stack((h_old, X_one_hot))

        hf, hf_cache = l.fc_forward(X, Wf, bf)
        hf, hf_sigm_cache = l.sigmoid_forward(hf)

        hi, hi_cache = l.fc_forward(X, Wi, bi)
        hi, hi_sigm_cache = l.sigmoid_forward(hi)

        ho, ho_cache = l.fc_forward(X, Wo, bo)
        ho, ho_sigm_cache = l.sigmoid_forward(ho)

        hc, hc_cache = l.fc_forward(X, Wc, bc)
        hc, hc_tanh_cache = l.tanh_forward(hc)

        c = hf * c_old + hi * hc
        c, c_tanh_cache = l.tanh_forward(c)

        h = ho * c

        cache = (
            X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache,
            ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache
        )

        if not train:
            y = util.softmax(h)
            return y, h, c

        return h, c, cache

    def backward(self, y_pred, y_train, dh_next, dc_next, cache):
        X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache, ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache = cache

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        dho = c * dy
        dho = l.sigmoid_backward(dho, ho_sigm_cache)
        dc = ho * dy
        dc = l.tanh_backward(dc, c_tanh_cache)

        dhf = c_old * dc
        dhf = l.sigmoid_backward(dhf, hf_sigm_cache)

        dhi = hc * dc
        dhi = l.sigmoid_backward(dhi, hi_sigm_cache)

        dhc = hi * dc
        dhc = l.tanh_backward(dhc, hc_tanh_cache)

        dXo, dWo, dbo = l.fc_backward(dho, ho_cache)
        dXc, dWc, dbc = l.fc_backward(dhc, hc_cache)
        dXi, dWi, dbi = l.fc_backward(dhi, hi_cache)
        dXf, dWf, dbf = l.fc_backward(dhf, hf_cache)

        dX = dXo + dXc + dXi + dXf
        dh_next = dX[:self.H]
        dc_next = hf * dc

        grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, bf=dbf, bi=dbi, bc=dbc, bo=dbo)

        return grad, dh_next, dc_next

    def train_step(self, X_train, y_train, h=None, c=None):
        hs = []
        caches = []
        loss = 0.

        if h is None:
            h = np.zeros((1, self.H))

        if c is None:
            c = np.zeros((1, self.D))

        # Forward
        for x, y in zip(X_train, y_train):
            h, c, cache = self.forward(x, h, c, train=True)
            loss += loss_fun.cross_entropy(self.model, h, y, lam=0)
            hs.append(h)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        dc_next = np.zeros((1, self.H))
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for state, cache, y_true in reversed(list(zip(hs, caches, y_train))):
            grad, dh_next, dc_next = self.backward(state, y_true, dh_next, dc_next, cache)

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, h, c

    def sample(self, X_seed, h, c, size=100):
        chars = [self.idx2char[X_seed]]
        idx_list = list(range(self.vocab_size))
        X = X_seed

        for _ in range(size - 1):
            prob, h, c = self.forward(X, h, c, train=False)
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.idx2char[idx])
            X = idx

        return ''.join(chars)

    def _init_model(self, D, C, H):
        Z = 2 * D

        self.model = dict(
            Wf=np.random.randn(Z, D) / np.sqrt(Z / 2.),
            Wi=np.random.randn(Z, D) / np.sqrt(Z / 2.),
            Wc=np.random.randn(Z, D) / np.sqrt(Z / 2.),
            Wo=np.random.randn(Z, D) / np.sqrt(Z / 2.),
            bf=np.zeros((1, D)),
            bi=np.zeros((1, D)),
            bc=np.zeros((1, D)),
            bo=np.zeros((1, D))
        )
