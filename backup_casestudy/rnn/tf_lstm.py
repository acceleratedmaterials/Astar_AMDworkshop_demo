# -*- coding: utf-8 -*-
"""
Tensorflow
Training samples: 170
Validation samples: 44
LSTM-layered RNN with 128 units
Optimizer: Adam
Epoch: 100
Loss: Cross Entropy
Activation function: Relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.5
Accuracy of Validation set: 63.64%
"""
from __future__ import division, print_function, absolute_import
import tflearn
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

net = tflearn.input_data([None, 10])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, 6, activation='relu')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.00005,
	loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
		batch_size=32, n_epoch=100)
		  
model.save('./saved/tf/lstm/model.tfl')