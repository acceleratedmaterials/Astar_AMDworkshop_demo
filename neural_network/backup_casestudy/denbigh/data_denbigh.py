denbigh_features = 3
import xlrd

def getDenbighData():
	workbook = xlrd.open_workbook('./denbigh_data_loader.xlsx', on_demand = True)
	worksheet = workbook.sheet_by_name('df')
	X = []
	Y = []
	for i in range(1, worksheet.nrows):
		features = [
				worksheet.cell(i, 1).value,
				worksheet.cell(i, 2).value,
				worksheet.cell(i, 3).value
		]
		category = worksheet.cell(i, 11).value
		X.append(features)
		Y.append(category)
	return X, Y
