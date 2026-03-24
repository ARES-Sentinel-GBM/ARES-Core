
import numpy as np
import pandas as pd
from scipy.integrate import odeint

class ARESSentinelV2:
    '''
    ARES v2: Enhanced Multi-Agent System for GBM.
    Optimized for EMA Regulatory Compliance (High Efficacy + High Safety).
    '''
    def __init__(self, initial_tumor=0.5, horizon_days=1095):
        self.t = np.linspace(0, horizon_days, horizon_days)
        self.y0 = [initial_tumor, 0.05, 0.01, 0.01, 1.0]

    def model_dynamics(self, y, t):
        T, B, V, F, H = y
        growth_trigger = 0.06 * (1 + 0.5 * np.sin(2 * np.pi * t / 500)) 
        
        # Potenziamento dei Gain dei biosensori (Versione v2)
        kill_V = 0.18 * V * T 
        comp_B = 0.07 * B * T 
        
        dTdt = (growth_trigger * T) - kill_V - comp_B - (0.06 * F * T)
        dBdt = 0.03 * B * (T / (T + 0.1)) - 0.01 * B
        dVdt = 0.10 * V * T - 0.03 * V 
        dFdt = 0.03 * F * T - 0.01 * F
        dHdt = 0.005 * (1.0 - H) - (0.0004 * T)
        return [dTdt, dBdt, dVdt, dFdt, dHdt]

    def simulate(self):
        sol = odeint(self.model_dynamics, self.y0, self.t)
        return pd.DataFrame(sol, columns=['Tumore', 'Batteri_Scout', 'Virus_Killer', 'Funghi_Jailer', 'Salute_Ospite'])
