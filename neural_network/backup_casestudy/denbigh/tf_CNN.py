"""
Tensorflow
denbigh
Training samples: 1600
Validation samples: 400
3 CNN layers each with 128 units, Average Pooling
Optimizer: Adam
Learning rate: 0.001
Epoch: 100
Mini batch size: 32
Activation function: relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.5, L2 for CNN layers and Cross Entropy for regression layer
Accuracy of Validation set: 95.25%
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, avg_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_denbigh import *
from utils.plot import *
import numpy as np

X, Y = getDenbighData()

n_epoch = 100
learning_rate = 0.001


X = pad_sequences(X, maxlen=10, value=0.)
Y = to_categorical(Y, 4)

network = input_data(shape=[None, 10], name='input')
network = tflearn.embedding(network, input_dim=100000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch1 = avg_pool_1d(branch1, 2)
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch2 = avg_pool_1d(branch2, 2)
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
branch3 = avg_pool_1d(branch3, 2)
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = dropout(network, 0.5)


network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=learning_rate,
                     loss='categorical_crossentropy', name='target')
					 
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch = n_epoch, shuffle=True, validation_set=0.2, show_metric=True, batch_size=32)
model.save('./model.tfl')

