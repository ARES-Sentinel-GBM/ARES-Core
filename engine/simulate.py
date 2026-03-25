 import numpy as np
from scipy.integrate import odeint

def tumor_model(y, t, params):
    T_sens, T_res = y

    r = params["r"]
    r_res = params["r_res"]
    K = params["K"]

    drug = params["drug"]

    growth = (1 - (T_sens + T_res)/K)

    dT_sens = r * T_sens * growth - drug * T_sens
    dT_res = r_res * T_res * growth

    return [dT_sens, dT_res]


def simulate_static(params, t):
    y0 = [0.1, 0.0]
    sol = odeint(tumor_model, y0, t, args=(params,))
    return sol[:,0] + sol[:,1]


def simulate_adaptive(params, t):
    y = np.array([0.1, 0.0])
    dt = t[1] - t[0]

    traj = []

    for _ in t:
        T_total = y[0] + y[1]

        # controllo a soglia
        drug = 0.8 if T_total > 0.5 else 0.2

        dydt = tumor_model(y, 0, {**params, "drug": drug})
        y = y + dt * np.array(dydt)
        y = np.clip(y, 0, None)

        traj.append(y.sum())

    return np.array(traj)


def simulate_rl(params, t):
    y = np.array([0.1, 0.0])
    dt = t[1] - t[0]

    traj = []

    for _ in t:
        T_total = y[0] + y[1]

        # policy RL (semplificata)
        if T_total > 0.6:
            drug = 1.0
        elif T_total > 0.4:
            drug = 0.5
        else:
            drug = 0.1

        dydt = tumor_model(y, 0, {**params, "drug": drug})
        y = y + dt * np.array(dydt)
        y = np.clip(y, 0, None)

        traj.append(y.sum())

    return np.array(traj)
