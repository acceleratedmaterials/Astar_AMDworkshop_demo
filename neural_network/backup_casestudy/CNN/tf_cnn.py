"""
Tensorflow
Training samples: 170
Validation samples: 44
CNN layers with 128 units
Optimizer: Adam
Epoch: 1000
Activation function: Relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.5, L2 for CNN layers and Cross Entropy for regression layer
Accuracy of Validation set: 66%
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_glass import *
trainX, trainY, testX, testY = getGlassData()


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=10, value=0.)
testX = pad_sequences(testX, maxlen=10, value=0.)
# # # Converting labels to binary vectors
trainY = to_categorical(trainY, 6)
testY = to_categorical(testY, 6)

network = input_data(shape=[None, 10], name='input')
network = tflearn.embedding(network, input_dim=1000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 1000, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)

model.save('./saved/tf/cnn/model.tfl')