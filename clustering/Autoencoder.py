# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 00:09:45 2018
@author: savitha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
from keras.models import Model#,load_model
#from keras.callbacks import ModelCheckpoint, TensorBoard


####### Data Setup
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 0
LABELS = ["Normal", "Fraud"]
df = pd.read_csv("denbigh_data_loader.csv")

### Data Stats ###
frauds = df[df.Class == 1]
normal = df[df.Class == 0]

#### Training Autoencoder
""" Normalize data between (0,1)"""
data =df
scaler = MinMaxScaler()
data[list(df)] = scaler.fit_transform(df[list(df)])
""" Split Data between Train and Test """
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values

####Model Building........................................................................
"""Hyperparameters"""
input_dim = X_train.shape[1]
encoding_dim = 4
nb_epoch = 100
batch_size = 20

""" Autoencoder architecture """
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
""" Training the autoencoder """
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = autoencoder.fit(X_train, X_train, epochs=nb_epoch, batch_size=batch_size,shuffle=True,validation_data=(X_test, X_test),verbose=1).history
### Predictions
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})


""" Metrics Report """
from sklearn.metrics import (auc,roc_curve)     
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();
    
    
