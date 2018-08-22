'''
Convert data from glass.data.txt into python-sensible data
Due to the small dataset of about 200 data, train data will contain about 190 samples 
and test data contain about 10 samples
'''
glass_features = 9
def getGlassData():
	f = open('data/glass/glass.data.txt', 'r')
	x = f.read().splitlines()
	trainX = []
	trainY = []
	testX = []
	testY = []
	i = 0
	for line in x:
		raw = line.split(',')
		#1st attribute is the ID, last attribute is the category
		category = int(raw[len(raw) - 1])
		#-- 4 vehicle_windows_non_float_processed is not in the dataset, thus to normalize
		# the output, category above 1 will be reduced by 1
		if category > 4:
			category -= 1
		
		#shift all category to the left by 1 for the vector to start at 0
		category -= 1
		data = [float(x) for x in raw[1:len(raw)-1]]
		if  i % 10 == 0 or i % 10 ==1:
			testX.append(data)
			testY.append(category)
		else:
			trainX.append(data)
			trainY.append(category)
		i += 1

	return trainX, trainY, testX, testY
