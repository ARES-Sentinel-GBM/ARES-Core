"""
modules/visualizer.py
=====================
Generazione visualizzazioni per il GBM Computational Pipeline.
Output: PNG ad alta risoluzione in /output/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Palette / stile ───────────────────────────────────────────────────────────
PALETTE = {
    "RTK/RAS/PI3K":        "#E63946",
    "Cell Cycle/p53":      "#457B9D",
    "Telomere/Epigenetics": "#2A9D8F",
    "Angiogenesis":        "#F4A261",
    "Metabolic":           "#8338EC",
    "Chromatin":           "#FB8500",
    "Transcription":       "#6D6875",
    "default":             "#ADB5BD",
}

def _pathway_color(pathway: str) -> str:
    return PALETTE.get(pathway, PALETTE["default"])

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
})


# ── 01 — Mutation Landscape ───────────────────────────────────────────────────
def plot_mutation_landscape(mut_df: pd.DataFrame) -> Path:
    top = mut_df.head(15).copy()
    colors = [_pathway_color(p) for p in top["pathway"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        top["gene"][::-1],
        top["frequency"][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.6,
        height=0.72,
    )

    # Etichette
    for bar, val in zip(bars, top["frequency"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8.5, color="#333")

    # Legenda pathway
    seen = {}
    for pathway, color in zip(top["pathway"], colors):
        if pathway not in seen:
            seen[pathway] = color
    patches = [mpatches.Patch(color=c, label=p) for p, c in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.7)

    ax.set_xlabel("Frequenza mutazionale (%)", fontsize=11)
    ax.set_title("GBM Mutation Landscape — TCGA-GBM (n=617, IDH-wt)\n"
                 "Brennan et al. Cell 2013", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 85)
    fig.tight_layout()

    out = OUTPUT_DIR / "01_mutation_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 02 — Target Scores ────────────────────────────────────────────────────────
def plot_target_scores(score_df: pd.DataFrame) -> Path:
    top = score_df.head(12).copy()
    colors = ["#2A9D8F" if d else "#ADB5BD" for d in top["druggable"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Subplot 1: Composite score ──
    ax1.barh(top["gene"][::-1], top["composite_score"][::-1],
             color=colors[::-1], edgecolor="white", height=0.72)
    ax1.axvline(0.5, color="#E63946", linestyle="--", alpha=0.6, label="Threshold 0.5")
    ax1.set_xlabel("Composite Score", fontsize=10)
    ax1.set_title("Target Priority Score\n(Druggable = verde)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)

    # ── Subplot 2: Scatter freq vs z-score ──
    sc = ax2.scatter(
        top["mutation_freq"],
        top["zscore"],
        s=top["composite_score"] * 400,
        c=["#E63946" if d else "#457B9D" for d in top["druggable"]],
        alpha=0.8, edgecolors="white", linewidths=0.8,
    )
    for _, row in top.iterrows():
        ax2.annotate(row["gene"],
                     (row["mutation_freq"], row["zscore"]),
                     fontsize=7.5, ha="left",
                     xytext=(3, 3), textcoords="offset points")

    ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Frequenza Mutazionale (%)", fontsize=10)
    ax2.set_ylabel("Expression Z-Score", fontsize=10)
    ax2.set_title("Mutazione vs Espressione\n(dimensione = composite score)", fontsize=11, fontweight="bold")

    red_p   = mpatches.Patch(color="#E63946", label="Druggable")
    blue_p  = mpatches.Patch(color="#457B9D", label="Non-druggable")
    ax2.legend(handles=[red_p, blue_p], fontsize=8)

    fig.suptitle("GBM Molecular Target Analysis", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUTPUT_DIR / "02_target_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 03 — PK Simulation ────────────────────────────────────────────────────────
def plot_pk_simulation(
    pk_df: pd.DataFrame,
    summary: dict,
    route_label: str = "FUS-NP",
) -> Path:
    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # PK curves full-width
    ax2 = fig.add_subplot(gs[1, 0])   # Effect
    ax3 = fig.add_subplot(gs[1, 1])   # Active drones

    t = pk_df["time_h"]

    # ── Plasma vs GBM ──
    ax1.plot(t, pk_df["c_plasma"], color="#457B9D", lw=2.2, label="Plasma")
    ax1.plot(t, pk_df["c_gbm"],    color="#E63946", lw=2.2, label="GBM tissue")
    ax1.fill_between(t, pk_df["c_gbm"], alpha=0.15, color="#E63946")

    # Annotazioni
    cmax_idx = pk_df["c_gbm"].idxmax()
    ax1.annotate(
        f"Cmax={summary['cmax_gbm']:.3f}\nTmax={summary['tmax_h']:.0f}h",
        xy=(pk_df.iloc[cmax_idx]["time_h"], pk_df.iloc[cmax_idx]["c_gbm"]),
        xytext=(pk_df.iloc[cmax_idx]["time_h"] + 4, pk_df.iloc[cmax_idx]["c_gbm"] * 0.9),
        fontsize=8.5, color="#E63946",
        arrowprops=dict(arrowstyle="->", color="#E63946", lw=1.2),
    )
    ax1.set_xlabel("Tempo (h)", fontsize=10)
    ax1.set_ylabel("Concentrazione (a.u.)", fontsize=10)
    ax1.set_title(
        f"Farmacocinetica Nanodrone — Rotta: {route_label}\n"
        f"BEE Penetration: {summary['bee_penetration']*100:.1f}%  |  "
        f"AUC_GBM: {summary['auc_gbm']:.3f}  |  T½: {summary['t_half_h']}h",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=9)

    # ── Effetto biologico ──
    ax2.plot(t, pk_df["effect"] * 100, color="#2A9D8F", lw=2)
    ax2.fill_between(t, pk_df["effect"] * 100, alpha=0.2, color="#2A9D8F")
    ax2.axhline(summary["peak_effect"] * 100, color="#F4A261",
                linestyle="--", lw=1.2, label=f"Peak {summary['peak_effect']*100:.1f}%")
    ax2.set_xlabel("Tempo (h)", fontsize=9)
    ax2.set_ylabel("Effetto (%)", fontsize=9)
    ax2.set_title("Effetto Biologico nel Tempo", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(pk_df["effect"] * 100) * 1.3)

    # ── Active drones ──
    ax3.plot(t, pk_df["active_drones"], color="#8338EC", lw=2)
    ax3.fill_between(t, pk_df["active_drones"], alpha=0.15, color="#8338EC")
    ax3.set_xlabel("Tempo (h)", fontsize=9)
    ax3.set_ylabel("Droni attivi (#)", fontsize=9)
    ax3.set_title(f"Flotta Attiva ({summary['total_drones']} droni totali)", fontsize=10, fontweight="bold")

    out = OUTPUT_DIR / "03_pk_simulation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 04 — Drug Comparison ─────────────────────────────────────────────────────
def plot_drug_comparison(comp_df: pd.DataFrame) -> Path:
    df = comp_df.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

    # ── Subplot 1: Efficacy score ranking ──
    colors = ["#E63946" if n else "#ADB5BD" for n in df["is_nanodrone"]]
    bars = ax1.barh(
        df["agent"][::-1],
        df["efficacy_score"][::-1],
        color=colors[::-1],
        edgecolor="white", height=0.72,
    )
    for bar, val in zip(bars, df["efficacy_score"][::-1]):
        ax1.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)

    ax1.set_xlabel("Efficacy Score Composito", fontsize=10)
    ax1.set_title("Ranking Farmaci GBM\n(Nano = rosso, Classici = grigio)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 0.90)
    ax1.axvline(0.5, color="#457B9D", linestyle="--", alpha=0.5, label="Soglia clinica")
    ax1.legend(fontsize=8)

    # ── Subplot 2: BEE vs Selectivity bubble ──
    ax2.scatter(
        df["bee_penetration"],
        df["selectivity"],
        s=df["efficacy_score"] * 600,
        c=["#E63946" if n else "#457B9D" for n in df["is_nanodrone"]],
        alpha=0.8, edgecolors="white", linewidths=0.8,
    )
    for _, row in df.iterrows():
        ax2.annotate(
            row["agent"].replace("Nano-", "N-").replace(" (BCNU)", "").replace(" (CCNU)",""),
            (row["bee_penetration"], row["selectivity"]),
            fontsize=7.5, xytext=(4, 3), textcoords="offset points",
        )

    ax2.axvline(0.5, color="#E63946", linestyle="--", alpha=0.5, label="BEE threshold 50%")
    ax2.axhline(0.5, color="#457B9D", linestyle="--", alpha=0.5, label="Sel threshold 50%")
    ax2.set_xlabel("BEE Penetration", fontsize=10)
    ax2.set_ylabel("Selettività", fontsize=10)
    ax2.set_title("BEE Penetration vs Selettività\n(dimensione = efficacy score)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("Nanodroni vs Farmaci Convenzionali — GBM Treatment Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = OUTPUT_DIR / "04_drug_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
