import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def printConfusionMatrix(conf_mat, num_classes=2, name='latest'):
	'''
	Print confusion matrix with heat map
	Params:
	- conf_mat: Confusion matrix
	- num_classes: Number of classes for label
	- name: Name of the saved file
	Return:
	- Void
	'''
	df_cm = pd.DataFrame(conf_mat, index = [i for i in range(num_classes)],
				columns = [i for i in range(num_classes)])
	# plt.figure(figsize = (10,7))
	sn.set(font_scale=1.1)#for label size
	sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
	plt.savefig("confusion_matrix/{}.png".format(name))
	plt.show()