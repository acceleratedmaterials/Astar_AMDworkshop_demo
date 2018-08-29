'''
Convert data from glass.data.txt into python-sensible data
Due to the small dataset of about 200 data, train data will contain about 190 samples 
and test data contain about 10 samples
'''
hydraulic_feature = 60

def getHydraulicCategory():
	f = open('./profile.txt', 'r')
	x = f.read().splitlines()
	y = []
	for line in x:
		y.append(int(line.split()[4]))
	return y

def getHydraulicFeature():
	f = open('./TS1.txt', 'r')
	x = f.read().splitlines()
	y = []
	for line in x:
		y.append([float(z) for z in line.split()])
	return y

def getHydraulicData():
	X = getHydraulicFeature()
	Y = getHydraulicCategory()

	trainX = []
	trainY = []
	testX = []
	testY = []
	i = 0
	for x, y in zip(X, Y):
		if  i % 10 == 0 or i % 10 ==1:
			testX.append(x)
			testY.append(y)
		else:
			trainX.append(x)
			trainY.append(y)
		i += 1

	return trainX, trainY, testX, testY
