'''
File: gradient_boosting_regression.py
Project: ML_workshop
File Created: Monday, 13th August 2018 12:20:26 am
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
'''

import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats


if __name__ == "__main__":
    # Set random seed
    np.random.seed(0)

    # Load data and do train-test Split
    df = pd.read_excel('./data/Concrete_Data.xls', sheet_name='Sheet1')
    X, y = df[df.columns[:-1]], df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Gradient boosting (decision tree) fit
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, y_train)
    y_hat_train = regressor.predict(X_train)
    y_hat_test = regressor.predict(X_test)

    # Plot predictions and feature importances
    utils.plot_predictions(
        y=[y_train, y_test],
        y_hat=[y_hat_train, y_hat_test],
        labels=['Train', 'Test'],
        save_path='./boosting_fit.pdf')
    utils.plot_feature_importances(
            importances=regressor.feature_importances_,
            columns=df.columns[:-1],
            save_path='./boosting_feat.pdf')

    # ####################################### #
    # Cross-Validation Hyper-parameter Tuning #
    # ####################################### #

    # Define random search space
    param_distributions = {
        'n_estimators': stats.randint(low=10, high=1000),
        'max_depth': stats.randint(low=2, high=6),
        'min_samples_split': stats.randint(low=2, high=5),
        'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01]}

    # Train a random CV regressor
    regressor_cv = RandomizedSearchCV(
        regressor, param_distributions=param_distributions,
        n_iter=50, verbose=1)
    regressor_cv.fit(X_train, y_train)
    y_hat_train = regressor_cv.predict(X_train)
    y_hat_test = regressor_cv.predict(X_test)

    # Plot predictions and feature importances
    utils.plot_predictions(
        y=[y_train, y_test],
        y_hat=[y_hat_train, y_hat_test],
        labels=['Train', 'Test'],
        save_path='./boosting_cv_fit.pdf')
    utils.plot_feature_importances(
            importances=regressor_cv.best_estimator_.feature_importances_,
            columns=df.columns[:-1],
            save_path='./boosting_feat.pdf')
