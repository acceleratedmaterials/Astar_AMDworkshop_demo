import xlrd
from collections import Counter


def removeDashes(arr):
	'''
	Some 0 values appear as '--' in excel
	Return:
	- list with '--' replaced by 0.0
	'''
	for idx, item in enumerate(arr):
		if item == '--':
			arr[idx] = 0.0

	return arr

def containsNegative(arr):
	'''Preprocessing data step
	Check if features contain a negative number.
	Return:
	- True if the features do
	_ False otherwise
	'''
	for idx, item in enumerate(arr):
		if item < 0:
			return True

	return False


def categorizePerformance(SH, selectivity):
	"""Categorize
    Categorize the performance of the cture of MOF/CORE based on SH and selectivity
	SH above 5 and selectivity above 15000 are considered ideal
	3 categories exist and 3 number from 1 -> 3 are assigned correspondingly.
	Return:
	- category number: 0 | 1 | 2
    """
	if SH <= 5 and selectivity <= 15000:
		return 0
	# This type doesn't exist
	# elif SH > 5 and selectivity <=15000: 
	# 	return 2
	elif SH <= 5 and selectivity >= 15000:
		return 1
	else:
		return 2

def gethMOFData():
	'''
	Extract hMOF data from the excel file and partion the data set with about 95% as training set and 5% as test set.
	Features are in order: Porosity, heat C1, heat C2, VSA, LCD
	Return:
	- trainX: training set features
	- trainY: training set catgories
	- testX: test set features
	- testY: test set categories
	'''
	workbook = xlrd.open_workbook('data/thiols/hMOF.xlsx', on_demand = True)
	worksheet = workbook.sheet_by_name('Sheet1')
	trainX = []
	trainY = []
	testX = []
	testY = []
	for i in range(1, worksheet.nrows):
		features = removeDashes([
				worksheet.cell(i, 7).value,
				worksheet.cell(i, 8).value,
				worksheet.cell(i, 9).value,
				worksheet.cell(i, 10).value,
				worksheet.cell(i, 12).value
		])
		category = categorizePerformance(worksheet.cell(i, 5).value, worksheet.cell(i, 6).value)
		if not containsNegative(features):
			if i % 8 == 0:
				testX.append(features)
				testY.append(category)
			else:
				trainX.append(features)
				trainY.append(category)

	return trainX, trainY, testX, testY

def getAllhMOFData():
	'''
	Extract hMOF data from the excel file without separating into train and test set.
	Features are in order: Porosity, heat C1, heat C2, VSA, LCD
	Return:
	- trainX: training set features
	- trainY: training set catgories
	- testX: test set features
	- testY: test set categories
	'''
	workbook = xlrd.open_workbook('data/thiols/hMOF.xlsx', on_demand = True)
	worksheet = workbook.sheet_by_name('Sheet1')
	X = []
	Y = []
	for i in range(1, worksheet.nrows):
		features = removeDashes([
				worksheet.cell(i, 7).value,
				worksheet.cell(i, 8).value,
				worksheet.cell(i, 9).value,
				worksheet.cell(i, 10).value,
				worksheet.cell(i, 12).value
		])
		category = categorizePerformance(worksheet.cell(i, 5).value, worksheet.cell(i, 6).value)
		if not containsNegative(features):
			X.append(features)
			Y.append(category)

	return X, Y

def getCOREData():
	'''
	Extract CoRE-MOFs data from the excel file and partion the data set with about 95% as training set and 5% as test set.
	Features are in order: Porosity, heat C1, heat C2, VSA, LCD
	Return:
	- trainX: training set features
	- trainY: training set catgories
	- testX: test set features
	- testY: test set categories
	'''
	workbook = xlrd.open_workbook('data/thiols/hMOF.xlsx', on_demand = True)
	worksheet = workbook.sheet_by_name('Sheet1')
	trainX = []
	trainY = []
	testX = []
	testY = []
	for i in range(1, worksheet.nrows):
		features = removeDashes([
				worksheet.cell(i, 21).value,
				worksheet.cell(i, 22).value,
				worksheet.cell(i, 24).value,
				worksheet.cell(i, 25).value,
				worksheet.cell(i, 26).value
			])
		category = categorizePerformance(worksheet.cell(i, 19).value,worksheet.cell(i, 20).value)
		if not containsNegative(features):
			if i % 18 == 0:
				testX.append(features)
				testY.append(category)
			else:
				trainX.append(features)
				trainY.append(category)

	return trainX, trainY, testX, testY


