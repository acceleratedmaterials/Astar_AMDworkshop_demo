from data.data_glass import *
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import random
import torch
import numpy as np
def getRandomData(n=20):
	trainX, trainY, testX, testY = getGlassData()
	trainX += testX
	trainY += testY
	idx = [random.randint(1, len(trainY) - 1) for x in range(n)]
	random_trainX = [trainX[i] for i in idx]
	random_trainY = [trainY[i] for i in idx]
	random_trainX = pad_sequences(random_trainX, maxlen=10, value=0.)
	return random_trainX, np.array(random_trainY)

def getTestInput():
	trainX, trainY, testX, testY = getGlassData()
	return pad_sequences(testX,  maxlen=10, value=0.)

def getTestOutput():
	trainX, trainY, testX, testY = getGlassData()
	return np.array(testY)

def getTrainData():
	batch_input_1, batch_y1 = getRandomData()
	batch_input_2, batch_y2 = getRandomData()
	batch_y = (batch_y1 == batch_y2).astype(float)
	data = [(torch.tensor(batch_input_1[i]), torch.tensor(batch_input_2[i]), batch_y[i]) for i in range(len(batch_y))]
	# return batch_input_1, batch_input_2, batch_y
	return data