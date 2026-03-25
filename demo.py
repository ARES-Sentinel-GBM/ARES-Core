"""
demo.py — ARES-Sentinel-GBM Interactive Demo
=============================================
Demo standalone con:
  1. Pipeline molecolare (mutation landscape + target scores)
  2. Simulazione PK nanodrone ottimale vs sub-ottimale
  3. Training agente RL (Q-Learning) con curva di apprendimento
  4. Top-5 configurazioni scoperte dall'agente

Esegui: python demo.py
Output: output/05_rl_training.png  +  output/06_rl_comparison.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from data.gbm_data import GBMDataLoader
from modules.target_analyzer import TargetAnalyzer
from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute, PKParameters
)
from modules.rl_optimizer import NanodronQAgent, QLearningConfig

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

W = 72  # line width per output


def sep(title=""):
    if title:
        pad = (W - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * W)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATI & TARGET
# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 1 — PROFILO MOLECOLARE GBM")
loader  = GBMDataLoader()
mut_df  = loader.load_mutations()
expr_df = loader.load_expression()
analyzer = TargetAnalyzer(mut_df, expr_df)
score_df = analyzer.score_targets()
sl_df    = analyzer.get_synthetic_lethality_map()

print(f"  Geni analizzati : {len(mut_df)}")
print(f"  Top target      : {score_df.iloc[0]['gene']} (score={score_df.iloc[0]['composite_score']:.3f})")
print(f"  SL high evidence: {len(sl_df[sl_df['evidence']=='HIGH'])} coppie")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PK: OTTIMALE vs SUB-OTTIMALE
# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 2 — SIMULAZIONE PK COMPARATA")

pk_params = PKParameters(bee_base=0.05, tumor_icp=25.0, tumor_ifp=22.0)

# Config ottimale (FUS + Transferrin)
fleet_opt = FleetConfig(
    n_sentinelle=6, n_decisori=1, n_attacco=3, n_lager=3,
    delivery_route=DeliveryRoute.FUS,
    drone_spec=DroneSpec(surface_coating="PEG+Transferrin",
                         immune_evasion=0.72, targeting_affinity=0.65)
)
# Config sub-ottimale (IV libera)
fleet_sub = FleetConfig(
    n_sentinelle=6, n_decisori=1, n_attacco=3, n_lager=3,
    delivery_route=DeliveryRoute.IV_FREE,
    drone_spec=DroneSpec(surface_coating="PEG",
                         immune_evasion=0.50, targeting_affinity=0.40)
)

sim_opt = NanodronePKSimulator(fleet_opt, pk_params)
sim_sub = NanodronePKSimulator(fleet_sub, pk_params)

pk_opt = sim_opt.simulate(dose=1.0, duration_h=72.0)
pk_sub = sim_sub.simulate(dose=1.0, duration_h=72.0)
sum_opt = sim_opt.pk_summary(pk_opt)
sum_sub = sim_sub.pk_summary(pk_sub)

print(f"  FUS+Transferrin → BEE={sum_opt['bee_penetration']*100:.1f}%  "
      f"Cmax={sum_opt['cmax_gbm']:.4f}  effect={sum_opt['peak_effect']*100:.1f}%")
print(f"  IV-free (PEG)   → BEE={sum_sub['bee_penetration']*100:.1f}%  "
      f"Cmax={sum_sub['cmax_gbm']:.4f}  effect={sum_sub['peak_effect']*100:.1f}%")
print(f"  Δ BEE: +{(sum_opt['bee_penetration']-sum_sub['bee_penetration'])*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 3 — RL TRAINING (Q-Learning, 600 episodi)")

cfg   = QLearningConfig(episodes=600, alpha=0.15, epsilon_decay=0.992)
agent = NanodronQAgent(cfg)

print("  Training in corso...", end="", flush=True)
history = agent.train()
print(" ✓")

best = agent.best_config()
top5 = agent.top_configs(5)

print(f"\n  Best config scoperta dall'agente:")
print(f"    Route   : {best['route']}")
print(f"    Coating : {best['coating']}")
print(f"    Fleet   : {best['fleet']}")
print(f"    BEE     : {best['bee']*100:.1f}%")
print(f"    Effect  : {best['peak_effect']*100:.2f}%")
print(f"    Reward  : {best['reward']:.4f}  (Q={best['q_value']:.4f})")

print(f"\n  Top-5 configurazioni:")
for _, row in top5.iterrows():
    print(f"    [{row['rank']}] {row['route']:<15} "
          f"{row['coating']:<18} "
          f"BEE={row['bee']*100:.0f}%  "
          f"rew={row['reward']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT: RL TRAINING CURVE
# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 4 — GENERAZIONE PLOT RL")

# Rolling average
window = 30
hist_r = history["reward"].rolling(window, min_periods=1).mean()
hist_e = history["epsilon"]
hist_b = history["bee"].rolling(window, min_periods=1).mean()
hist_q = history["q_max"]

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

kw = dict(facecolor="#0D1117")

# ── Plot 1: Reward curve ──
ax1 = fig.add_subplot(gs[0, :], **kw)
ax1.set_facecolor("#161B22")
ax1.plot(history["episode"], history["reward"],
         color="#30363D", lw=0.6, alpha=0.5)
ax1.plot(history["episode"], hist_r,
         color="#58A6FF", lw=2.2, label=f"Reward (rolling {window}ep)")
ax1.fill_between(history["episode"], hist_r,
                 alpha=0.15, color="#58A6FF")

best_ep = history.loc[history["reward"].idxmax(), "episode"]
best_rew = history["reward"].max()
ax1.axvline(best_ep, color="#F78166", lw=1.2, linestyle="--", alpha=0.8)
ax1.annotate(f"Best\n{best_rew:.4f}",
             xy=(best_ep, best_rew), xytext=(best_ep+20, best_rew*0.95),
             color="#F78166", fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="#F78166"))

ax1.set_title("Q-Learning — Curva di Apprendimento (Reward per Episodio)",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Episodio", color="#8B949E"); ax1.set_ylabel("Reward", color="#8B949E")
ax1.tick_params(colors="#8B949E"); ax1.spines[:].set_color("#30363D")
ax1.legend(fontsize=9, facecolor="#161B22", labelcolor="white")
ax1.grid(True, alpha=0.2, color="#30363D")

# ── Plot 2: Epsilon decay ──
ax2 = fig.add_subplot(gs[1, 0], **kw)
ax2.set_facecolor("#161B22")
ax2.plot(history["episode"], hist_e, color="#3FB950", lw=2)
ax2.fill_between(history["episode"], hist_e, alpha=0.15, color="#3FB950")
ax2.set_title("Epsilon Decay (Esplorazione → Sfruttamento)",
              color="white", fontsize=10, fontweight="bold")
ax2.set_xlabel("Episodio", color="#8B949E")
ax2.set_ylabel("ε (esplorazione)", color="#8B949E")
ax2.tick_params(colors="#8B949E"); ax2.spines[:].set_color("#30363D")
ax2.grid(True, alpha=0.2, color="#30363D")

# ── Plot 3: BEE rolling + Q-max ──
ax3 = fig.add_subplot(gs[1, 1], **kw)
ax3.set_facecolor("#161B22")
ax3_r = ax3.twinx()
ax3_r.set_facecolor("#161B22")

ax3.plot(history["episode"], hist_b * 100,
         color="#F8C53A", lw=2, label="BEE avg %")
ax3_r.plot(history["episode"], hist_q,
           color="#D2A8FF", lw=1.5, linestyle="--", label="Q-max")

ax3.set_title("BEE Penetration scoperta & Q-max",
              color="white", fontsize=10, fontweight="bold")
ax3.set_xlabel("Episodio", color="#8B949E")
ax3.set_ylabel("BEE (%)", color="#F8C53A")
ax3_r.set_ylabel("Q-max", color="#D2A8FF")
ax3.tick_params(colors="#8B949E"); ax3_r.tick_params(colors="#D2A8FF")
ax3.spines[:].set_color("#30363D"); ax3_r.spines[:].set_color("#30363D")
ax3.grid(True, alpha=0.2, color="#30363D")

lines  = [plt.Line2D([0],[0], color="#F8C53A", lw=2),
          plt.Line2D([0],[0], color="#D2A8FF", lw=1.5, ls="--")]
labels = ["BEE avg %", "Q-max"]
ax3.legend(lines, labels, fontsize=8, facecolor="#161B22", labelcolor="white")

fig.suptitle("ARES-Sentinel-GBM — RL Optimizer: Q-Learning Fleet Optimization",
             color="white", fontsize=14, fontweight="bold", y=1.01)

out_rl = OUTPUT / "05_rl_training.png"
fig.savefig(out_rl, dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig)
print(f"  ✓ {out_rl.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT: CONFRONTO OTTIMALE vs SUB-OTTIMALE (PK curves)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 6))
fig2.patch.set_facecolor("#0D1117")

titles = [
    ("FUS + PEG+Transferrin (RL-ottimale)", "#58A6FF", pk_opt, sum_opt),
    ("IV-free + PEG (sub-ottimale)",        "#F78166", pk_sub, sum_sub),
]
for ax, (title, color, pk_df, summary) in zip(axes, titles):
    ax.set_facecolor("#161B22")
    ax.spines[:].set_color("#30363D")
    t = pk_df["time_h"]
    ax.plot(t, pk_df["c_plasma"], color="#8B949E", lw=1.6, label="Plasma", linestyle="--")
    ax.plot(t, pk_df["c_gbm"],    color=color,     lw=2.2, label="GBM tissue")
    ax.fill_between(t, pk_df["c_gbm"], alpha=0.18, color=color)

    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Tempo (h)", color="#8B949E"); ax.set_ylabel("Concentrazione (a.u.)", color="#8B949E")
    ax.tick_params(colors="#8B949E")
    ax.legend(fontsize=8.5, facecolor="#161B22", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#30363D")

    bee_txt = f"BEE: {summary['bee_penetration']*100:.1f}%\nCmax: {summary['cmax_gbm']:.4f}"
    ax.text(0.98, 0.97, bee_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=color,
            bbox=dict(facecolor="#0D1117", edgecolor=color, alpha=0.7, boxstyle="round,pad=0.4"))

fig2.suptitle("PK Comparison — RL-ottimale vs Sub-ottimale",
              color="white", fontsize=13, fontweight="bold")
fig2.tight_layout()

out_cmp = OUTPUT / "06_rl_comparison.png"
fig2.savefig(out_cmp, dpi=150, bbox_inches="tight", facecolor="#0D1117")
plt.close(fig2)
print(f"  ✓ {out_cmp.name}")

sep("DEMO COMPLETATO")
print(f"\nOutput generati:")
print(f"  {out_rl}")
print(f"  {out_cmp}")
print(f"\nAgente RL — best config: {best['route']} + {best['coating']}")
print(f"Reward ottimale: {best['reward']:.4f}  |  BEE: {best['bee']*100:.1f}%\n")
