import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, matthews_corrcoef, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def create_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, activation='relu',
                       optimizer='Adam', reg_term=0,  dropout=0,
                       verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'activation', 'optimizer',
             'reg_term', 'dropout',
             'verbose']
    values = [hidden_layers, learning_rate, epochs, batch_size, activation, optimizer, reg_term,
              dropout, verbose]
    hparams = dict(zip(names, values))
    return hparams

class DNN_classifer:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim=fl.n_classes
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss='categorical_crossentropy')

    def create_DNN_model(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='sigmoid'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features=fl.features_c_norm_a
        training_labels=fl.labels_hot
        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def eval(self, eval_fl):
        eval_start=time.time()
        features=eval_fl.features_c_norm_a
        labels=eval_fl.labels
        labels_hot=eval_fl.labels_hot
        predictions=self.model.predict(features)
        predictions_class = [predicted_labels_hot.index(max(predicted_labels_hot)) for predicted_labels_hot in
                             np.ndarray.tolist(predictions)]
        # Calculating metrics
        acc = accuracy_score(labels, predictions_class)
        ce = log_loss(labels_hot, predictions)
        cm = confusion_matrix(labels, predictions_class)
        try:
            f1s = f1_score(labels, predictions_class)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predictions_class, average='micro')  # Must use micro averaging for multi-class
        mcc=matthews_corrcoef(labels,predictions_class)

        eval_end=time.time()
        print('eval run time : {}'.format(eval_end-eval_start))
        return predictions_class, acc, ce, cm, f1s, mcc

class DNN:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim= fl.labels_dim
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss='mse')

    def create_DNN_model(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='linear'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features=fl.features_c_norm_a
        training_labels=fl.labels
        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()

    def eval(self, eval_fl, mode='c'):
        features=eval_fl.features_c_norm_a
        predictions = self.model.predict(features)
        if mode=='c' or mode=='r':
            labels=eval_fl.labels
            # Calculating metrics
            mse=mean_squared_error(labels, predictions)
            return predictions, mse
        elif mode=='p':
            return predictions


