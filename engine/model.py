import numpy as np

def tumor_step(y, drug, params):
    T_sens, T_res = y

    r = params["r"]
    r_res = params["r_res"]
    K = params["K"]

    T_total = T_sens + T_res
    growth = (1 - T_total / K)

    dT_sens = r * T_sens * growth - drug * T_sens
    dT_res = r_res * T_res * growth

    return np.array([dT_sens, dT_res])
