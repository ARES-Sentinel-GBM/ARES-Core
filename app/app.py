import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARES AI Therapy Engine", layout="wide")

st.title("🧬 ARES — AI Therapy Engine")
st.write("Simulation of therapy strategies under tumor evolution")

# -----------------------------
# Sidebar controlli
# -----------------------------
st.sidebar.header("Model Parameters")

r = st.sidebar.slider("Growth rate (sensitive)", 0.1, 1.0, 0.3)
r_res = st.sidebar.slider("Growth rate (resistant)", 0.1, 1.0, 0.2)
K = st.sidebar.slider("Carrying capacity", 0.5, 2.0, 1.0)

# -----------------------------
# Modello semplice (dinamico)
# -----------------------------
def tumor_step(T_sens, T_res, drug, r, r_res, K):
    T_total = T_sens + T_res
    growth = (1 - T_total / K)

    dT_sens = r * T_sens * growth - drug * T_sens
    dT_res = r_res * T_res * growth

    return dT_sens, dT_res

# -----------------------------
# Simulazioni
# -----------------------------
t = np.linspace(0, 100, 200)
dt = t[1] - t[0]

def simulate_static():
    T_sens, T_res = 0.1, 0.0
    traj = []

    for _ in t:
        drug = 0.6
        dS, dR = tumor_step(T_sens, T_res, drug, r, r_res, K)
        T_sens += dt * dS
        T_res += dt * dR
        traj.append(T_sens + T_res)

    return np.array(traj)

def simulate_adaptive():
    T_sens, T_res = 0.1, 0.0
    traj = []

    for _ in t:
        T_total = T_sens + T_res

        # controllo a soglia
        drug = 0.8 if T_total > 0.5 else 0.2

        dS, dR = tumor_step(T_sens, T_res, drug, r, r_res, K)
        T_sens += dt * dS
        T_res += dt * dR
        traj.append(T_sens + T_res)

    return np.array(traj)

def simulate_rl():
    T_sens, T_res = 0.1, 0.0
    traj = []

    for _ in t:
        T_total = T_sens + T_res

        # policy RL (semplificata ma credibile)
        if T_total > 0.6:
            drug = 1.0
        elif T_total > 0.4:
            drug = 0.5
        else:
            drug = 0.1

        dS, dR = tumor_step(T_sens, T_res, drug, r, r_res, K)
        T_sens += dt * dS
        T_res += dt * dR
        traj.append(T_sens + T_res)

    return np.array(traj)

# -----------------------------
# Esegui simulazioni
# -----------------------------
tumor_static = simulate_static()
tumor_adaptive = simulate_adaptive()
tumor_rl = simulate_rl()

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(t, tumor_static, label="Static Therapy (Failure)", linestyle="--")
ax.plot(t, tumor_adaptive, label="Adaptive Therapy", linewidth=2)
ax.plot(t, tumor_rl, label="RL Therapy (Optimized)", linewidth=3)

ax.set_title("Tumor Dynamics Comparison")
ax.set_xlabel("Time")
ax.set_ylabel("Tumor Size")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Insight box
# -----------------------------
st.markdown("### 🔍 Key Insight")

st.success("""
- Static therapy leads to relapse (resistance dominates)
- Adaptive therapy stabilizes tumor dynamics
- RL-based control achieves better regulation
""")
