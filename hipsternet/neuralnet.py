import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.regularization as reg
import hipsternet.layer as l
import hipsternet.constant as c


class NeuralNet(object):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent'):
        self._init_model(D, C, H)

        if loss not in ('cross_ent', 'hinge'):
            raise Exception('Loss function must be either "cross_ent" or "hinge"!')

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss

    def train_step(self, X_train, y_train):
        """
        Single training step over minibatch: forward, loss, backprop
        """
        y_pred, cache = self.forward(X_train, train=True)

        if self.loss == 'cross_ent':
            loss = loss_fun.cross_entropy(self.model, y_pred, y_train, self.lam)
        elif self.loss == 'hinge':
            loss = loss_fun.hinge_loss(self.model, y_pred, y_train, self.lam)

        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def forward(self, X, train=False):
        m = X.shape[0]

        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
        gamma1, gamma2 = self.model['gamma1'], self.model['gamma2']
        beta1, beta2 = self.model['beta1'], self.model['beta2']
        bn1_mean, bn2_mean = self.bn_caches['bn1_mean'], self.bn_caches['bn2_mean']
        bn1_var, bn2_var = self.bn_caches['bn1_var'], self.bn_caches['bn2_var']

        u1, u2 = None, None
        bn1_cache, bn2_cache = None, None

        # Input to hidden
        h1 = X @ W1 + b1

        # BatchNorm
        if train:
            h1, bn1_cache, run_mean, run_var = l.batchnorm_forward(h1, gamma1, beta1, (bn1_mean, bn1_var))
            self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'] = run_mean, run_var
        else:
            h1 = (h1 - bn1_mean) / np.sqrt(bn1_var + c.eps)
            h1 = gamma1 * h1 + beta1

        # ReLU
        h1[h1 < 0] = 0

        if train:
            # Dropout
            u1 = np.random.binomial(1, self.p_dropout, size=h1.shape) / self.p_dropout
            h1 *= u1

        # Hidden to hidden
        h2 = h1 @ W2 + b2

        # BatchNorm
        if train:
            h2, bn2_cache, run_mean, run_var = l.batchnorm_forward(h2, gamma2, beta2, (bn2_mean, bn2_var))
            self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'] = run_mean, run_var
        else:
            h2 = (h2 - bn2_mean) / np.sqrt(bn2_var + c.eps)
            h2 = gamma2 * h2 + beta2

        # ReLU
        h2[h2 < 0] = 0

        if train:
            # Dropout
            u2 = np.random.binomial(1, self.p_dropout, size=h2.shape) / self.p_dropout
            h2 *= u2

        # Hidden to output
        score = h2 @ W3 + b3

        return score, (X, h1, h2, u1, u2, bn1_cache, bn2_cache)

    def backward(self, y_pred, y_train, cache):
        X, h1, h2, u1, u2, bn1_cache, bn2_cache = cache

        # Output layer
        if self.loss == 'cross_ent':
            grad_y = loss_fun.dcross_entropy(y_pred, y_train)
        elif self.loss == 'hinge':
            grad_y = loss_fun.dhinge_loss(y_pred, y_train)

        # Third layer
        dh2, dW3, db3 = l.fc_backward(grad_y, h2, self.model['W3'], lam=self.lam)
        dh2 = l.dropout_backward(dh2, u2)
        dh2, dgamma2, dbeta2 = l.batchnorm_backward(dh2, bn2_cache)

        # Second layer
        dh1, dW2, db2 = l.fc_backward(dh2, h1, self.model['W2'], lam=self.lam)
        dh1 = l.dropout_backward(dh1, u1)
        dh1, dgamma1, dbeta1 = l.batchnorm_backward(dh1, bn1_cache)

        # First layer
        _, dW1, db1 = l.fc_backward(dh1, X, self.model['W1'], lam=self.lam, input_layer=True)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
            gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
        )

        return grad

    def predict_proba(self, X):
        score, _ = self.forward(X, False)
        return l.softmax(score)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

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
