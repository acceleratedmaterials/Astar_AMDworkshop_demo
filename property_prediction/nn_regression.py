'''
File: nn_regression.py
Project: ML_workshop
File Created: Monday, 13th August 2018 12:48:42 am
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
'''

import numpy as np
import pandas as pd
import logging
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


if __name__ == "__main__":
    # Logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Set random seed
    np.random.seed(0)

    # Load data and do train-test Split
    df = pd.read_excel('./data/Concrete_Data.xls', sheet_name='Sheet1')
    X, y = df[df.columns[:-1]], df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Scale inputs
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP regressor fit
    regressor = MLPRegressor(
        hidden_layer_sizes=[256, 128, 64], max_iter=1000)
    regressor.fit(X_train_scaled, y_train)
    y_hat_train = regressor.predict(X_train_scaled)
    y_hat_test = regressor.predict(X_test_scaled)

    # Plot predictions
    utils.plot_predictions(
        y=[y_train, y_test],
        y_hat=[y_hat_train, y_hat_test],
        labels=['Train', 'Test'],
        save_path='./nn_fit.pdf')

    # ############################ #
    # Illustration of over-fitting #
    # ############################ #

    # Data split with validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1
    )
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # MLP regressor fit
    # For demo purposes, we are going to fit the NN on a smaller dataset
    # instead of going for a much bigger network (and training for a long time)
    X_train, y_train = X_train[:50], y_train[:50]
    X_train_scaled, X_valid_scaled, X_test_scaled = \
        map(scaler.transform, [X_train, X_valid, X_test])
    regressor = MLPRegressor(
        hidden_layer_sizes=[256, 128, 64], max_iter=100, warm_start=True,
        solver='sgd', alpha=0, momentum=0,
        learning_rate='adaptive', learning_rate_init=1e-3)

    # Train and log the losses
    n_iter = 50
    train_losses, valid_losses, test_losses = [], [], []
    for n in range(n_iter):
        regressor.fit(X_train_scaled, y_train)
        train_losses.append(
            utils.rmse(regressor.predict(X_train_scaled), y_train))
        valid_losses.append(
            utils.rmse(regressor.predict(X_valid_scaled), y_valid))
        test_losses.append(
            utils.rmse(regressor.predict(X_test_scaled), y_test))
        logging.info(
            'Iteration %d | Train loss %.3f | Valid loss %.3f | Test loss %.3f'
            % (n, train_losses[-1], valid_losses[-1], test_losses[-1]))

    y_hat_train = regressor.predict(X_train_scaled)
    y_hat_test = regressor.predict(X_test_scaled)

    # Plot predictions
    utils.plot_predictions(
        y=[y_train, y_test],
        y_hat=[y_hat_train, y_hat_test],
        labels=['Train', 'Test'],
        save_path='./nn_overfit.pdf')

    # Plot training curves
    utils.plot_training_curves(
        losses=[train_losses, valid_losses, test_losses],
        labels=['Train', 'Validation', 'Test'],
        save_path='./nn_overfit_training_curves.pdf')
