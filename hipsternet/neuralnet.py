import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.layer as l


class NeuralNet(object):

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
        if loss not in ('cross_ent', 'hinge'):
            raise Exception('Loss function must be either "cross_ent" or "hinge"!')

        if nonlin not in ('relu', 'lrelu', 'sigmoid', 'tanh'):
            raise Exception('Nonlinearity must be either "relu", "lrelu", "sigmoid", or "tanh"!')

        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss
        self.forward_nonlin = NeuralNet.forward_nonlins[nonlin]
        self.backward_nonlin = NeuralNet.backward_nonlins[nonlin]

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
        gamma1, gamma2 = self.model['gamma1'], self.model['gamma2']
        beta1, beta2 = self.model['beta1'], self.model['beta2']

        u1, u2 = None, None
        bn1_cache, bn2_cache = None, None

        # First layer
        h1 = l.fc_forward(X, self.model['W1'], self.model['b1'])
        bn1_cache = (self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'])
        h1, bn1_cache, run_mean, run_var = l.bn_forward(h1, gamma1, beta1, bn1_cache, train=train)
        h1 = self.forward_nonlin(h1)

        self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'] = run_mean, run_var

        if train:
            h1, u1 = l.dropout_forward(h1, self.p_dropout)

        # Second layer
        h2 = l.fc_forward(h1, self.model['W2'], self.model['b2'])
        bn2_cache = (self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'])
        h2, bn2_cache, run_mean, run_var = l.bn_forward(h2, gamma2, beta2, bn2_cache, train=train)
        h2 = self.forward_nonlin(h2)

        self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'] = run_mean, run_var

        if train:
            h2, u2 = l.dropout_forward(h2, self.p_dropout)

        # Third layer
        score = l.fc_forward(h2, self.model['W3'], self.model['b3'])

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
        dh2 = self.backward_nonlin(dh2, h2)
        dh2 = l.dropout_backward(dh2, u2)
        dh2, dgamma2, dbeta2 = l.bn_backward(dh2, bn2_cache)

        # Second layer
        dh1, dW2, db2 = l.fc_backward(dh2, h1, self.model['W2'], lam=self.lam)
        dh1 = self.backward_nonlin(dh1, h1)
        dh1 = l.dropout_backward(dh1, u1)
        dh1, dgamma1, dbeta1 = l.bn_backward(dh1, bn1_cache)

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
