# -*- coding: utf-8 -*-
'''
Framework: Tensorflow
Training samples: 1600
Validation samples: 400
RNN with 128 units
Optimizer: Adam
Epoch: 100
Loss: Cross Entropy
Activation function: Relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.8
Accuracy of Validation set: 95%
'''
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_denbigh import *

X, Y = getDenbighData()

#Hyperparams
neurons_num = 128    # Number of neurons in the RNN layer
keep_prob = 0.5	 # Keep probability for the drop-out regularization
learning_rate = 0.001 # Learning rate for mini-batch SGD
batch_size = 32		 # Batch size
n_epoch = 100       # Number of epoch


#Data preprocessing/ Converting data to vector for the 
X = pad_sequences(X, maxlen=5, value=0.)
Y = to_categorical(Y, 2)

#Build the network
net = tflearn.input_data([None, 5])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.simple_rnn(net, neurons_num, dropout=keep_prob)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
	loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, validation_set=0.2, show_metric=True,
		batch_size=batch_size, n_epoch=n_epoch)
		  
model.save('./saved/tf/rnn_hydraulic/model.tfl')