from data.data_glass import *
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import random
import numpy as np
def getRandomData(n=100, test=0):
	trainX, trainY, testX, testY = getGlassData()
	if (test == 1):
		return pad_sequences(testX, maxlen=10, value=0.), np.array(testY)
	else:
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

def getTestData():
	batch_input_1, batch_y1 = getRandomData(test=1)
	batch_input_2, batch_y2 = getRandomData(n=44)
	batch_y = (batch_y1 != batch_y2).astype(float)
	return batch_input_1, batch_input_2, batch_y
