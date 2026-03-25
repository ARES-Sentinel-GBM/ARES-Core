import numpy as np
from engine.model import tumor_step


def simulate_static(params, t):
    y = np.array([0.1, 0.0])
    dt = t[1] - t[0]
    traj = []

    for _ in t:
        drug = 0.6
        y += dt * tumor_step(y, drug, params)
        y = np.clip(y, 0, None)
        traj.append(y.sum())

    return np.array(traj)


def simulate_adaptive(params, t):
    y = np.array([0.1, 0.0])
    dt = t[1] - t[0]
    traj = []

    for _ in t:
        T_total = y.sum()
        drug = 0.8 if T_total > 0.5 else 0.2

        y += dt * tumor_step(y, drug, params)
        y = np.clip(y, 0, None)
        traj.append(y.sum())

    return np.array(traj)


def simulate_rl(params, t):
    y = np.array([0.1, 0.0])
    dt = t[1] - t[0]
    traj = []

    for _ in t:
        T_total = y.sum()

        # policy RL (placeholder ma modulare)
        if T_total > 0.6:
            drug = 1.0
        elif T_total > 0.4:
            drug = 0.5
        else:
            drug = 0.1

        y += dt * tumor_step(y, drug, params)
        y = np.clip(y, 0, None)
        traj.append(y.sum())

    return np.array(traj)
