'''
File: inverse_design.py
Project: ML_workshop
File Created: Monday, 13th August 2018 1:44:39 am
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
'''

import pandas as pd
import numpy as np
import logging
import utils
from sklearn.ensemble import GradientBoostingRegressor
from pyswarm import pso


if __name__ == "__main__":

    # Logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Set random seed
    np.random.seed(0)

    # Load entire dataset
    df = pd.read_excel('./data/Concrete_Data.xls', sheet_name='Sheet1')
    X, y = df[df.columns[:-1]], df[df.columns[-1]]

    # Params are found by RandomSearchCV
    # We omit running this again and simply quote results from a run
    params = {'learning_rate': 0.05,
              'max_depth': 2,
              'min_samples_split': 2,
              'n_estimators': 994}
    regressor = GradientBoostingRegressor()
    regressor.set_params(**params)
    regressor.fit(X, y)

    # Define bounds, objectives and constraints
    upper_bounds = np.percentile(X, 75, axis=0) * 1.5
    lower_bounds = np.percentile(X, 25, axis=0) * 0.5

    def objective(X):
        """
        We want to minimize
            Blast Furnace Slag, Fly Ash, Superplasticizer
        """
        return X[1]**2 + X[2]**2 + X[4]**2

    def constraints(X):
        """
        We want to following constraints:
            1. Compressive strength >= 70 MPa
            2. Water <= 150 kg / m^3
            3. Age <= 30 days
        """
        predicted_strength = regressor.predict(X.reshape(1, -1))
        cons_str_lower = predicted_strength - 70
        cons_water_upper = 150 - X[3]
        cons_age_upper = 30 - X[-1]
        return [cons_str_lower, cons_water_upper, cons_age_upper]

    # Log info
    logging.info(objective.__doc__)
    logging.info(constraints.__doc__)

    # Run PSO 5 times with some generic hyper-parameters
    X_opts = []
    n_runs = 5
    for n in range(n_runs):
        logging.info('Run %d' % (n))
        X_opt, _ = pso(
            objective, lower_bounds, upper_bounds, f_ieqcons=constraints,
            swarmsize=100, maxiter=200)
        X_opts.append(X_opt)
    X_opts = np.asarray(X_opts)
    y_hat_opts = regressor.predict(X_opts).reshape(-1, 1)
    data_opt = np.concatenate([X_opts, y_hat_opts], axis=1)
    df_predict = pd.DataFrame(columns=df.columns, data=data_opt)
    logging.info('Particle swarm optimization completed...\n\n')

    # Display Results
    for n in range(5):
        print('Run: %d' % n)
        print(df_predict.loc[n], '\n\n')

    # Compare with unseen data
    df_unseen = pd.read_excel(
        './data/Concrete_Data_unseen.xls', sheet_name='Sheet1')
    print('Compare with unseen data:')
    print(df_unseen.loc[0])
