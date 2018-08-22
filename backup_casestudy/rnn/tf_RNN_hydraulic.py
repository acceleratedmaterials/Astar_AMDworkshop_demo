# -*- coding: utf-8 -*-
'''
Condition monitoring of hydraulic systems
=========================================

Abtract: The data set addresses the condition assessment of a hydraulic test rig based on multi sensor data. Four fault types are superimposed with several severity grades impeding selective quantification.

Source:
Creator: ZeMA gGmbH, Eschberger Weg 46, 66121 Saarbr¸cken
Contact: t.schneider@zema.de, s.klein@zema.de, m.bastuck@lmt.uni-saarland.de, info@lmt.uni-saarland.de

Data Type: Multivariate, Time-Series
Task: Classification, Regression
Attribute Type: Categorical, Real
Area: CS/Engineering
Format Type: Matrix
Does your data set contain missing values? No

Number of Instances: 2205

Number of Attributes: 43680 (8x60 (1 Hz) + 2x600 (10 Hz) + 7x6000 (100 Hz))

Relevant Information:
The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit which are connected via the oil tank [1], [2]. The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied. 

Attribute Information:
The data set contains raw process sensor data (i.e. without feature extraction) which are structured as matrices (tab-delimited) with the rows representing the cycles and the columns the data points within a cycle. The sensors involved are:
Sensor		Physical quantity		Unit		Sampling rate
PS1		Pressure			bar		100 Hz
PS2		Pressure			bar		100 Hz
PS3		Pressure			bar		100 Hz
PS4		Pressure			bar		100 Hz
PS5		Pressure			bar		100 Hz
PS6		Pressure			bar		100 Hz
EPS1		Motor power			W		100 Hz
FS1		Volume flow			l/min		10 Hz
FS2		Volume flow			l/min		10 Hz
TS1		Temperature			∞C		1 Hz
TS2		Temperature			∞C		1 Hz
TS3		Temperature			∞C		1 Hz
TS4		Temperature			∞C		1 Hz
VS1		Vibration			mm/s		1 Hz
CE		Cooling efficiency (virtual)	%		1 Hz
CP		Cooling power (virtual)		kW		1 Hz
SE		Efficiency factor		%		1 Hz

The target condition values are cycle-wise annotated in ëprofile.txtë (tab-delimited). As before, the row number represents the cycle number. The columns are

1: Cooler condition / %:
	3: close to total failure
	20: reduced effifiency
	100: full efficiency

2: Valve condition / %:
	100: optimal switching behavior
	90: small lag
	80: severe lag
	73: close to total failure

3: Internal pump leakage:
	0: no leakage
	1: weak leakage
	2: severe leakage

4: Hydraulic accumulator / bar:
	130: optimal pressure
	115: slightly reduced pressure
	100: severely reduced pressure
	90: close to total failure

5: stable flag:
	0: conditions were stable
	1: static conditions might not have been reached yet

Relevant Papers:
[1] Nikolai Helwig, Eliseo Pignanelli, Andreas Sch¸tze, ëCondition Monitoring of a Complex Hydraulic System Using Multivariate Statisticsí, in Proc. I2MTC-2015 - 2015 IEEE International Instrumentation and Measurement Technology Conference, paper PPS1-39, Pisa, Italy, May 11-14, 2015, doi: 10.1109/I2MTC.2015.7151267.
[2] N. Helwig, A. Sch¸tze, ëDetecting and compensating sensor faults in a hydraulic condition monitoring systemí, in Proc. SENSOR 2015 - 17th International Conference on Sensors and Measurement Technology, oral presentation D8.1, Nuremberg, Germany, May 19-21, 2015, doi: 10.5162/sensor2015/D8.1.
[3] Tizian Schneider, Nikolai Helwig, Andreas Sch¸tze, ëAutomatic feature extraction and selection for classification of cyclical time series dataí, tm - Technisches Messen (2017), 84(3), 198ñ206, doi: 10.1515/teme-2016-0072.

Citation Requests:
Nikolai Helwig, Eliseo Pignanelli, Andreas Sch¸tze, ëCondition Monitoring of a Complex Hydraulic System Using Multivariate Statisticsí, in Proc. I2MTC-2015 - 2015 IEEE International Instrumentation and Measurement Technology Conference, paper PPS1-39, Pisa, Italy, May 11-14, 2015, doi: 10.1109/I2MTC.2015.7151267.


Framework: Tensorflow
Training samples: 1763
Validation samples: 442
RNN with 32 units
Optimizer: Adam
Epoch: 35
Loss: Cross Entropy
Activation function: Relu for network and Soft-max for regression
Regularization: Drop-out, keep_prob = 0.8
Accuracy of Validation set: 80%
'''
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from data.data_hydraulic import *

trainX, trainY, testX, testY = getHydraulicData()

#Hyperparams
neurons_num = 128    # Number of neurons in the RNN layer
keep_prob = 0.7 	 # Keep probability for the drop-out regularization
learning_rate = 0.01 # Learning rate for mini-batch SGD
batch_size = 32		 # Batch size
n_epoch = 1000       # Number of epoch


#Data preprocessing/ Converting data to vector for the 
trainX = pad_sequences(trainX, maxlen=60, value=0.)
testX = pad_sequences(testX, maxlen=60, value=0.)
trainY = to_categorical(trainY, 2)
testY = to_categorical(testY, 2)

#Build the network
net = tflearn.input_data([None, 60])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.simple_rnn(net, neurons_num, dropout=keep_prob)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
	loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
		batch_size=batch_size, n_epoch=n_epoch)
		  
model.save('./saved/tf/rnn_hydraulic/model.tfl')