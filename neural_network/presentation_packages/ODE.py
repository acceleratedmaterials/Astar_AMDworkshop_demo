import numpy as np
from scipy.integrate.odepack import odeint

def denbigh_rxn2(input_features, A=[.2, 0.01, 0.005, 0.005], E=[7000, 3000, 3000, 100]):
    '''
    Returns output concentration for a batch reactor based on the denbigh reaction.
    Reaction parameters has been tuned to ensure variability within default trans range
    :param input_features: Features class object containing features information. Last 2 column must be time
    and temperature.
    First n columns represent n species initial concentration
    :param A: Pre-Exponent factor for Arrhenius Equation
    :param E: Activation Energy
    :param plot_mode: Plot conc vs time graph for last set of features if == True
    :return: Label class containing output concentrations of A,R,T,S,U
    '''
    numel_row = input_features.shape[0]
    numel_col = input_features.shape[1]
    conc = input_features[:, :numel_col - 2]
    c_out = []

    def reaction(c, t, T, A, E):
        # Rate equations for Denbigh reaction from Chemical Reaction Engineering, Levenspiel 3ed Chapter 8, page 194
        [Ca, Cr, Ct, Cs, Cu] = c
        [k1, k2, k3, k4] = A * np.exp(-np.array(E) / (8.314 * T))
        [dCadt, dCrdt, dCtdt, dCsdt, dCudt] = [-(k1 + k2) * Ca ** 2,
                                               k1 * Ca ** 2 - (k3 + k4) * Cr ** 0.5,
                                               k2 * Ca ** 2,
                                               k3 * Cr ** 0.5,
                                               k4 * Cr ** 0.5]
        return [dCadt, dCrdt, dCtdt, dCsdt, dCudt]

    if numel_col < 7:
        # If input_features has less than 7 columns, means not all species have non zero inital conc. Must add zero cols
        zeros = np.zeros((numel_row, 7 - numel_col))
        input_features = np.concatenate((conc, zeros, input_features[:, -2:]), axis=1)

    for i in range(numel_row):
        c0 = input_features[i, :-2]
        t = np.linspace(0, input_features[i, -2], 2000)
        T = input_features[i, -1]
        c = odeint(reaction, c0, t, args=(T, A, E))
        c_out.append(c[-1, :])

    c_out = np.array(c_out)  # To convert list of numpy array to n x m numpy array
    return c_out
