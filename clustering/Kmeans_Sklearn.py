from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score 
# Importing the dataset
data = pd.read_csv('denbigh_data_loader.csv')
print("Input Data and Shape")
print(data.shape)
data.head()
# Getting the Input
X=data.values
y = data['Class']        
# Number of clusters
kmeans = KMeans(n_clusters=2)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print("sklearn")
print(centroids) # From sci-kit learn
print(accuracy_score(y, labels))
## Visualize
plt.figure(1),plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);

from sklearn.metrics import (auc,roc_curve)     
fpr, tpr, thresholds = roc_curve(y,labels)
roc_auc = auc(fpr, tpr)
plt.figure(2),plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,2],[0,2],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.201])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();