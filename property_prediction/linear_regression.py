'''
File: linear_regression.py
Project: ML_workshop
File Created: Monday, 13th August 2018 12:05:41 am
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
'''

import numpy as np
import pandas as pd
import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    # Set random seed
    np.random.seed(0)

    # Load data and do train-test Split
    df = pd.read_excel('./data/Concrete_Data.xls', sheet_name='Sheet1')
    X, y = df[df.columns[:-1]], df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Linear regression fit
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_hat_train = regressor.predict(X_train)
    y_hat_test = regressor.predict(X_test)

    # Plot predictions
    utils.plot_predictions(
        y=[y_train, y_test],
        y_hat=[y_hat_train, y_hat_test],
        labels=['Train', 'Test'],
        save_path='./linreg_fit.pdf')
