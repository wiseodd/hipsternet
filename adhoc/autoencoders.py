from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

X_train, y_train = mnist.train.images, mnist.train.labels
X_val, y_val = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels


def autoencoder(X, loss='l2', lam=0.):
    X = X.reshape(X.shape[0], -1)
    M, N = X.shape

    inputs = Input(shape=(N,))
    h = Dense(64, activation='sigmoid')(inputs)
    outputs = Dense(N)(h)

    model = Model(input=inputs, output=outputs)
    loss = 'mae' if loss == 'l1' else 'mse'

    model.compile(optimizer='adam', loss=loss)
    model.fit(X, X, batch_size=64, nb_epoch=5)

    return model


def sparse_autoencoder(X, lam=1e-3):
    X = X.reshape(X.shape[0], -1)
    M, N = X.shape

    inputs = Input(shape=(N,))
    h = Dense(64, activation='sigmoid')(inputs)
    outputs = Dense(N)(h)

    model = Model(input=inputs, output=outputs)

    def sparse_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        sparse = 1e-3 * K.mean(K.abs(h))
        return mse + sparse

    model.compile(optimizer='adam', loss=sparse_loss)
    model.fit(X, X, batch_size=64, nb_epoch=5)

    return model


def conv_autoencoder(X):
    X = X.reshape(X.shape[0], 28, 28, 1)

    inputs = Input(shape=(28, 28, 1))
    h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(inputs)
    h = MaxPooling2D((2, 2))(h)
    h = Conv2D(8, 3, 3, activation='relu', border_mode='same')(h)

    # 7x7x8 = 392
    encoded = MaxPooling2D((2, 2))(h)

    h = UpSampling2D((2, 2))(encoded)
    h = Conv2D(8, 3, 3, activation='relu', border_mode='same')(h)
    h = UpSampling2D((2, 2))(h)
    outputs = Conv2D(1, 3, 3, activation='relu', border_mode='same')(h)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, batch_size=64, nb_epoch=5)

    return model


def contractive_autoencoder(X, lam=1e-3):
    X = X.reshape(X.shape[0], -1)
    M, N = X.shape

    inputs = Input(shape=(N,))
    h = Dense(64, activation='sigmoid', name='encoded')(inputs)
    outputs = Dense(N, activation='linear')(h)

    model = Model(input=inputs, output=outputs)

    def contractive_loss(y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        W = K.variable(value=model.get_layer('encoded').get_weights()[0])
        jacobian = (h * (1 - h)) * (K.sum(W, axis=0))
        contractive = lam * K.sum(K.square(jacobian))
        return mse + contractive

    model.compile(optimizer='adam', loss=contractive_loss)
    model.fit(X, X, batch_size=64, nb_epoch=5)

    return model


if __name__ == '__main__':
    model = contractive_autoencoder(X_train)

    idxs = np.random.randint(0, X_test.shape[0], size=5)
    X_recons = model.predict(X_test[idxs])

    for X_recon in X_recons:
        plt.imshow(X_recon.reshape(28, 28), cmap='Greys_r')
        plt.show()
