# hipsternet
All the hipster things in Neural Net in a single repo: hipster optimization algorithms, hispter regularizations, everything!

Note, things will be added over time, so not all the hipsterest things will be here immediately. Also don't use this for your production code: use this to study and learn new things in the realm of Neural Net, Deep Net, Deep Learning, whatever.

## What's in it?

#### Network Architectures

1. Convolutional Net
2. Feed Forward Net
3. Recurrent Net
4. LSTM Net
5. GRU Net

#### Optimization algorithms

1. SGD
2. Momentum SGD
3. Nesterov Momentum
4. Adagrad
5. RMSprop
6. Adam

#### Loss functions

1. Cross Entropy
2. Hinge Loss
3. Squared Loss
4. L1 Regression
5. L2 Regression

#### Regularization

1. Dropout
2. Your usual L1 and L2 regularization

#### Nonlinearities

1. ReLU
2. leaky ReLU
3. sigmoid
4. tanh

#### Hipster techniques

1. BatchNorm
2. Xavier weight initialization

#### Pooling

1. Max pooling
2. Average pooling

## How to run this?

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate hipsternet`
4. [Optional] To install Tensorflow: `chmod +x tensorflow.sh; ./tensorflow.sh`
5. Do things with the code if you want to
6. To run the example:
  1. `python run_mnist.py {ff|cnn}`; `cnn` for convnet model, `ff` for the feed forward model
  2. `python run_rnn.py {rnn|lstm|gru}`; `rnn` for vanilla RNN model, `lstm` for LSTM net model, `gru` for GRU net model
7. Just close the terminal if you done (or `source deactivate`, not a fan though)

## What can I do with this?

Do anything you want. I licensed this with Unlicense License <http://unlicense.org>, as I need to take a break of using WTFPL license.
