# -*- coding: utf-8 -*-
"""
Tensorflow
Training samples: 170
Validation samples: 44
Bidirectional RNN with LSTM forward and LSTM backward, each layer with 128 units
Optimizer: Adam
Epoch: 1000
Loss: Cross Entropy
Activation function: Softmax
Regularization: Drop-out, keep_prob = 0.5
Accuracy of Training set: 70%
Accuracy of Validation set: 60%
"""
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_glass import *
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression


trainX, trainY, testX, testY = getGlassData()


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=10, value=0.)
testX = pad_sequences(testX, maxlen=10, value=0.)
# # # Converting labels to binary vectors
trainY = to_categorical(trainY, 6)
testY = to_categorical(testY, 6)

# Network building
net = input_data(shape=[None, 10])
net = embedding(net, input_dim=1000, output_dim=128)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = dropout(net, 0.5)
net = fully_connected(net, 6, activation='softmax')
net = regression(net, optimizer='adam', loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, clip_gradients=0.,tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=(testX, testY), 
	show_metric=True, batch_size=32, n_epoch=1000)
		  
model.save('./saved/tf/bidirectionRNN/model.tfl')