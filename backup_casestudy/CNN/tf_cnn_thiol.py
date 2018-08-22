"""
Tensorflow
hMOF classification
Training samples: 24761
Validation samples: 1304
3 CNN layers each with 128 units, Average Pooling
Optimizer: Adam
Learning rate: 0.001
Epoch: 100
Mini batch size: 256
Activation function: relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.5, L2 for CNN layers and Cross Entropy for regression layer
Accuracy of Validation set: 97.24%
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, avg_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_thiol import *
from utils.plot import *
import numpy as np

X, Y = getAllhMOFData()

n_epoch = 100
learning_rate = 0.001


X = pad_sequences(X, maxlen=10, value=0.)
Y = to_categorical(Y, 3)

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


network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=learning_rate,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit(X, Y, n_epoch = n_epoch, shuffle=True, validation_set=0.05, show_metric=True, batch_size=256)

model.load('./saved/tf/cnn_thiol/model.tfl')
# model.save('./saved/tf/cnn_thiol/model.tfl')

# General confusion matrix
_, _, testX, testY = gethMOFData()
testX = pad_sequences(testX, maxlen=10, value=0.)
y_ = model.predict(testX)
y_ = np.argmax(y_, axis=1)
with tf.Session().as_default(): 
    conf_mat = tf.confusion_matrix(labels=testY, predictions=y_, num_classes=3).eval()
    printConfusionMatrix(conf_mat, num_classes=3, name='Thiol_conf_matrix')

