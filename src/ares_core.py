import numpy as np
import pandas as pd
from scipy.integrate import odeint

class ARESSentinel:
    def __init__(self, initial_tumor=0.5, horizon_days=1095):
        self.t = np.linspace(0, horizon_days, horizon_days)
        self.y0 = [initial_tumor, 0.05, 0.01, 0.01, 1.0]

    def model_dynamics(self, y, t):
        T, B, V, F, H = y
        growth_trigger = 0.03 * (1 + 0.5 * np.sin(2 * np.pi * t / 500))
        kill_V = 0.12 * V * T
        comp_B = 0.04 * B * T
        jail_F = 0.06 * F * T
        dTdt = (growth_trigger * T) - kill_V - comp_B - jail_F
        dBdt = 0.02 * B * (T / (T + 0.1)) - 0.01 * B
        dVdt = 0.08 * V * T - 0.03 * V
        dFdt = 0.03 * F * T - 0.01 * F
        dHdt = 0.005 * (1.0 - H) - (0.0005 * T)
        return [dTdt, dBdt, dVdt, dFdt, dHdt]

    def simulate(self):
        sol = odeint(self.model_dynamics, self.y0, self.t)
        return pd.DataFrame(sol, columns=['Tumore', 'Batteri_Scout', 'Virus_Killer', 'Funghi_Jailer', 'Salute_Ospite'])

# Esecuzione rapida per rigenerare i file mancanti
ares = ARESSentinel()
df = ares.simulate()
df.to_csv('/content/ARES_final_simulation.csv', index=False)
print("✅ File ares_core.py e CSV creati correttamente in /content")
