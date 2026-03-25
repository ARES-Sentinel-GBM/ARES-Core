"""
app.py — ARES-Sentinel-GBM Streamlit Dashboard
================================================
UI interattivo per esplorazione computazionale del GBM.

Esegui: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARES-Sentinel-GBM",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace; }

.stApp { background: #0A0E1A; color: #E6EDF3; }

section[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58A6FF; }
.metric-value { font-size: 2rem; font-weight: 700; color: #58A6FF; }
.metric-label { font-size: 0.78rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
.metric-delta { font-size: 0.85rem; color: #3FB950; margin-top: 4px; }

.section-header {
    color: #58A6FF;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
}
.tag {
    display: inline-block;
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #9CA3AF;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.tag-green { border-color: #3FB950; color: #3FB950; background: #0A1F10; }
.tag-red   { border-color: #F85149; color: #F85149; background: #1F0A0A; }
.tag-blue  { border-color: #58A6FF; color: #58A6FF; background: #0A1020; }

.stButton > button {
    background: linear-gradient(135deg, #1C6FA8 0%, #0D4F8A 100%);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 20px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

div[data-testid="stTabs"] button {
    color: #8B949E;
    font-weight: 500;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58A6FF;
    border-bottom-color: #58A6FF;
}

.stSlider > div { color: #E6EDF3; }
hr { border-color: #21262D; }
</style>
""", unsafe_allow_html=True)

# ── Imports pipeline ─────────────────────────────────────────────────────────
from data.gbm_data import GBMDataLoader
from modules.target_analyzer import TargetAnalyzer, BEE_PENETRATION
from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute, PKParameters
)
from modules.rl_optimizer import NanodronQAgent, QLearningConfig

# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    loader  = GBMDataLoader()
    mut_df  = loader.load_mutations()
    expr_df = loader.load_expression()
    drug_df = loader.load_drugs()
    return mut_df, expr_df, drug_df

@st.cache_data
def compute_scores(mut_key, expr_key):
    mut_df, expr_df, _ = load_data()
    analyzer = TargetAnalyzer(mut_df, expr_df)
    score_df   = analyzer.score_targets()
    pathway_df = analyzer.rank_pathways()
    sl_df      = analyzer.get_synthetic_lethality_map()
    return score_df, pathway_df, sl_df

# ── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🧬")
with col_title:
    st.markdown("""
    <h1 style='color:#E6EDF3;font-size:1.8rem;font-weight:700;margin-bottom:0'>
        ARES-Sentinel-GBM
    </h1>
    <p style='color:#8B949E;font-size:0.9rem;margin-top:2px'>
        Adaptive Reinforcement Ecosystem — Nanodrone PK & Target Analysis for Glioblastoma
        &nbsp;|&nbsp;
        <span style='color:#58A6FF'>TCGA-GBM n=617</span> &nbsp;·&nbsp;
        <span style='color:#3FB950'>IDH-wildtype</span> &nbsp;·&nbsp;
        <span style='color:#F8C53A'>Brennan et al. Cell 2013</span>
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#21262D;margin:10px 0 20px 0'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Fleet Configuration")
    st.markdown("---")

    route_names = {
        "FUS-NP (Focused Ultrasound)": DeliveryRoute.FUS,
        "Transferrin-NP (RMT)":        DeliveryRoute.TRANSFERRIN,
        "CED (Convection-Enhanced)":   DeliveryRoute.CED,
        "Exosome":                     DeliveryRoute.EXOSOME,
        "NP-PLGA":                     DeliveryRoute.NP_PLGA,
        "Liposome":                    DeliveryRoute.LIPOSOME,
        "IV-free":                     DeliveryRoute.IV_FREE,
    }
    route_sel    = st.selectbox("🚀 Delivery Route", list(route_names.keys()), index=0)
    delivery_rte = route_names[route_sel]

    coating_sel = st.selectbox("🧪 Surface Coating", [
        "PEG+Transferrin", "Transferrin", "PEG", "Angiopep2", "Lactoferrin"
    ], index=0)

    st.markdown("**Fleet SDAL**")
    col_s, col_d = st.columns(2)
    col_a, col_l = st.columns(2)
    n_s = col_s.number_input("Sentinelle", 1, 10, 6)
    n_d = col_d.number_input("Decisori",   1,  5, 1)
    n_a = col_a.number_input("Attacco",    1, 10, 3)
    n_l = col_l.number_input("Lager",      1, 10, 3)

    st.markdown("**PK Parameters**")
    dose_val = st.slider("Dose (a.u.)", 0.25, 3.0, 1.0, 0.25)
    icp_val  = st.slider("ICP (mmHg)",  10,   40,  25)
    ifp_val  = st.slider("IFP (mmHg)",  5,    35,  22)
    dur_h    = st.slider("Durata sim. (h)", 24, 96, 72, 24)

    st.markdown("---")
    st.markdown("**RL Optimizer**")
    rl_episodes = st.slider("Episodi Q-Learning", 100, 1000, 400, 100)
    run_rl      = st.button("🤖 Lancia RL Training")

# ── Load data ─────────────────────────────────────────────────────────────────
mut_df, expr_df, drug_df = load_data()
score_df, pathway_df, sl_df = compute_scores("v1", "v1")

# ── KPI Row ───────────────────────────────────────────────────────────────────
spec = DroneSpec(
    surface_coating=coating_sel,
    immune_evasion=0.72,
    targeting_affinity=0.65,
)
fleet = FleetConfig(
    n_sentinelle=n_s, n_decisori=n_d, n_attacco=n_a, n_lager=n_l,
    delivery_route=delivery_rte, drone_spec=spec,
)
pk_params = PKParameters(bee_base=0.05, tumor_icp=icp_val, tumor_ifp=ifp_val)
sim = NanodronePKSimulator(fleet, pk_params)
pk_df   = sim.simulate(dose=dose_val, duration_h=dur_h)
summary = sim.pk_summary(pk_df)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

def kpi_card(col, value, label, delta=None, color="#58A6FF"):
    with col:
        delta_html = f'<div class="metric-delta">▲ {delta}</div>' if delta else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

kpi_card(kpi1, f"{summary['bee_penetration']*100:.1f}%", "BEE Penetration", color="#58A6FF")
kpi_card(kpi2, f"{summary['cmax_gbm']:.4f}",             "Cmax in GBM",     color="#F8C53A")
kpi_card(kpi3, f"{summary['tmax_h']:.0f}h",              "Tmax",            color="#3FB950")
kpi_card(kpi4, f"{summary['t_half_h']}h",                "T½ stimato",      color="#D2A8FF")
kpi_card(kpi5, f"{fleet.total_drones}",                  f"Droni Totali ({n_s}S/{n_d}D/{n_a}A/{n_l}L)", color="#F78166")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Mutation Landscape",
    "🎯 Target Analysis",
    "💉 PK Simulation",
    "⚖️ Drug Comparison",
    "🤖 RL Optimizer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Mutation Landscape
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">GBM Mutation Landscape — TCGA (n=617, IDH-wt)</div>', unsafe_allow_html=True)

    PALETTE = {
        "RTK/RAS/PI3K": "#E63946", "Cell Cycle/p53": "#457B9D",
        "Telomere/Epigenetics": "#2A9D8F", "Angiogenesis": "#F4A261",
        "Metabolic": "#8338EC", "Chromatin": "#FB8500",
        "Transcription": "#6D6875", "default": "#ADB5BD",
    }

    top = mut_df.head(15)
    colors = [PALETTE.get(p, PALETTE["default"]) for p in top["pathway"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
    bars = ax.barh(top["gene"][::-1], top["frequency"][::-1],
                   color=colors[::-1], edgecolor="#0D1117", height=0.72)
    for bar, val in zip(bars, top["frequency"][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8.5, color="#8B949E")
    ax.set_xlabel("Frequenza Mutazionale (%)", color="#8B949E", fontsize=10)
    ax.set_title("Top 15 Geni Mutati in GBM", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#8B949E"); ax.spines[:].set_color("#21262D")
    ax.set_xlim(0, 85)
    ax.grid(True, axis="x", alpha=0.2, color="#30363D")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col_t, col_p = st.columns(2)
    with col_t:
        st.markdown("**Top mutazioni**")
        st.dataframe(
            mut_df[["gene","frequency","pathway","alteration_type"]].head(10)
            .rename(columns={"gene":"Gene","frequency":"Freq (%)","pathway":"Pathway","alteration_type":"Tipo"}),
            use_container_width=True, height=300
        )
    with col_p:
        st.markdown("**Pathway ranking**")
        st.dataframe(
            pathway_df[["pathway","mean_score","druggable_genes","max_freq"]]
            .rename(columns={"pathway":"Pathway","mean_score":"Score","druggable_genes":"Druggable","max_freq":"Max Freq %"}),
            use_container_width=True, height=300
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Target Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Analisi Target Molecolari & Letalità Sintetica</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        top12 = score_df.head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        bar_colors = ["#2A9D8F" if d else "#374151" for d in top12["druggable"]]
        bars = ax.barh(top12["gene"][::-1], top12["composite_score"][::-1],
                       color=bar_colors[::-1], edgecolor="#0D1117", height=0.72)
        ax.axvline(0.5, color="#F78166", linestyle="--", lw=1.2, alpha=0.7, label="Threshold 0.5")
        ax.set_xlabel("Composite Score", color="#8B949E")
        ax.set_title("Target Priority (verde = druggable)", color="white", fontweight="bold")
        ax.tick_params(colors="#8B949E"); ax.spines[:].set_color("#21262D")
        ax.grid(True, axis="x", alpha=0.2, color="#30363D")
        ax.legend(fontsize=8, facecolor="#161B22", labelcolor="white")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_r:
        st.markdown("**Top 8 Target Druggabili**")
        display_df = score_df[score_df["druggable"]].head(8)[
            ["priority_rank","gene","composite_score","mutation_freq","pathway"]
        ].rename(columns={
            "priority_rank":"#","gene":"Gene","composite_score":"Score",
            "mutation_freq":"Freq %","pathway":"Pathway"
        })
        st.dataframe(display_df, use_container_width=True, height=280)

        st.markdown("**Letalità Sintetica (HIGH)**")
        sl_high = sl_df[sl_df["evidence"]=="HIGH"][
            ["mutated_gene","target_gene","mut_frequency","reference"]
        ].rename(columns={"mutated_gene":"Mutazione","target_gene":"Target","mut_frequency":"Freq %","reference":"Fonte"})
        st.dataframe(sl_high, use_container_width=True, height=200)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PK Simulation
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Simulazione Farmacocinetica Nanodrone</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BEE Penetration", f"{summary['bee_penetration']*100:.1f}%")
    c2.metric("Cmax GBM",        f"{summary['cmax_gbm']:.4f}")
    c3.metric("AUC GBM",         f"{summary['auc_gbm']:.4f}")
    c4.metric("Peak Effect",     f"{summary['peak_effect']*100:.2f}%")

    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    # PK full
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161B22")
    t = pk_df["time_h"]
    ax1.plot(t, pk_df["c_plasma"], color="#8B949E", lw=1.8, linestyle="--", label="Plasma")
    ax1.plot(t, pk_df["c_gbm"],    color="#58A6FF", lw=2.2, label="GBM tissue")
    ax1.fill_between(t, pk_df["c_gbm"], alpha=0.15, color="#58A6FF")
    cmax_idx = pk_df["c_gbm"].idxmax()
    ax1.axvline(pk_df.iloc[cmax_idx]["time_h"], color="#F8C53A", lw=1, linestyle=":", alpha=0.7)
    ax1.set_title(f"PK — {route_sel} | Coating: {coating_sel} | Dose: {dose_val}",
                  color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Tempo (h)", color="#8B949E"); ax1.set_ylabel("Concentrazione (a.u.)", color="#8B949E")
    ax1.tick_params(colors="#8B949E"); ax1.spines[:].set_color("#21262D")
    ax1.legend(fontsize=9, facecolor="#161B22", labelcolor="white")
    ax1.grid(True, alpha=0.2, color="#30363D")

    # Effect
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161B22")
    ax2.plot(t, pk_df["effect"]*100, color="#3FB950", lw=2)
    ax2.fill_between(t, pk_df["effect"]*100, alpha=0.18, color="#3FB950")
    ax2.set_title("Effetto Biologico (%)", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Tempo (h)", color="#8B949E"); ax2.set_ylabel("Effetto (%)", color="#8B949E")
    ax2.tick_params(colors="#8B949E"); ax2.spines[:].set_color("#21262D")
    ax2.grid(True, alpha=0.2, color="#30363D")

    # Drones
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161B22")
    ax3.plot(t, pk_df["active_drones"], color="#D2A8FF", lw=2)
    ax3.fill_between(t, pk_df["active_drones"], alpha=0.15, color="#D2A8FF")
    ax3.set_title(f"Droni Attivi ({fleet.total_drones} totali)", color="white", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Tempo (h)", color="#8B949E"); ax3.set_ylabel("#", color="#8B949E")
    ax3.tick_params(colors="#8B949E"); ax3.spines[:].set_color("#21262D")
    ax3.grid(True, alpha=0.2, color="#30363D")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # BEE strategy table
    st.markdown("**BEE Strategy Matrix**")
    bee_df = pd.DataFrame([
        {"Metodo": k, "BEE %": f"{v*100:.0f}%",
         "Invasivo": "⚠️ Sì" if v > 0.85 else "✅ No",
         "Consigliato": "✅" if v >= 0.60 else ""}
        for k, v in BEE_PENETRATION.items()
    ])
    st.dataframe(bee_df, use_container_width=True, height=250)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Drug Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Nanodroni vs Farmaci Convenzionali — GBM</div>', unsafe_allow_html=True)

    from modules.nanodrone_sim import DrugComparison
    comp_df = DrugComparison().compare()

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        bar_colors = ["#F78166" if n else "#374151" for n in comp_df["is_nanodrone"]]
        bars = ax.barh(comp_df["agent"][::-1], comp_df["efficacy_score"][::-1],
                       color=bar_colors[::-1], edgecolor="#0D1117", height=0.72)
        for bar, val in zip(bars, comp_df["efficacy_score"][::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color="#8B949E")
        ax.axvline(0.5, color="#58A6FF", lw=1.2, linestyle="--", alpha=0.6)
        ax.set_xlabel("Efficacy Score", color="#8B949E")
        ax.set_title("Ranking Agenti (Nano=arancio, Classici=grigio)", color="white", fontweight="bold")
        ax.tick_params(colors="#8B949E"); ax.spines[:].set_color("#21262D")
        ax.set_xlim(0, 0.95)
        ax.grid(True, axis="x", alpha=0.2, color="#30363D")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_r:
        st.markdown("**Tabella comparativa**")
        disp = comp_df[["agent","efficacy_score","bee_penetration","selectivity","is_nanodrone"]].copy()
        disp["bee_penetration"] = (disp["bee_penetration"]*100).round(0).astype(int).astype(str) + "%"
        disp["is_nanodrone"] = disp["is_nanodrone"].map({True: "🧬 Nano", False: "💊 Farm"})
        disp.columns = ["Agente","Score","BEE","Sel","Tipo"]
        st.dataframe(disp, use_container_width=True, height=380)

        best_nano = comp_df[comp_df["is_nanodrone"]].iloc[0]
        best_farm = comp_df[~comp_df["is_nanodrone"]].iloc[0]
        delta_pct = (best_nano["efficacy_score"] - best_farm["efficacy_score"]) / best_farm["efficacy_score"] * 100
        st.success(f"🧬 **{best_nano['agent']}** vs 💊 **{best_farm['agent']}**  \n"
                   f"Score +{delta_pct:.1f}% | BEE +{(best_nano['bee_penetration']-best_farm['bee_penetration'])*100:.0f}pp")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RL Optimizer
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Q-Learning Fleet Optimizer</div>', unsafe_allow_html=True)

    st.markdown("""
    L'agente **Q-Learning** esplora lo spazio di configurazione della flotta nanodrone:
    dosi, route di delivery, coating superficiale e composizione SDAL.
    Il reward è: `0.50 × BEE + 0.35 × peak_effect − 0.15 × tossicità_sistemica`
    """)

    if run_rl or "rl_history" not in st.session_state:
        if run_rl:
            with st.spinner(f"Training Q-Learning ({rl_episodes} episodi)..."):
                cfg   = QLearningConfig(episodes=rl_episodes, alpha=0.15, epsilon_decay=0.992)
                agent = NanodronQAgent(cfg)
                history = agent.train()
                best_rl = agent.best_config()
                top5_rl = agent.top_configs(5)
                st.session_state["rl_history"] = history
                st.session_state["rl_best"]    = best_rl
                st.session_state["rl_top5"]    = top5_rl
            st.success("✅ Training completato!")
        else:
            st.info("👆 Premi **Lancia RL Training** nella sidebar per avviare l'ottimizzazione.")
            st.stop()

    history = st.session_state["rl_history"]
    best_rl = st.session_state["rl_best"]
    top5_rl = st.session_state["rl_top5"]

    # KPI best
    bk1, bk2, bk3, bk4 = st.columns(4)
    bk1.metric("Best Reward",    f"{best_rl['reward']:.4f}")
    bk2.metric("BEE ottimale",   f"{best_rl['bee']*100:.1f}%")
    bk3.metric("Route ottimale", best_rl['route'])
    bk4.metric("Coating",        best_rl['coating'])

    # Plot training
    window = max(10, rl_episodes // 30)
    hist_r = history["reward"].rolling(window, min_periods=1).mean()
    hist_e = history["epsilon"]
    hist_b = history["bee"].rolling(window, min_periods=1).mean()
    hist_q = history["q_max"]

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161B22")
    ax1.plot(history["episode"], history["reward"], color="#30363D", lw=0.5, alpha=0.4)
    ax1.plot(history["episode"], hist_r, color="#58A6FF", lw=2.2, label=f"Rolling {window}ep")
    ax1.fill_between(history["episode"], hist_r, alpha=0.15, color="#58A6FF")
    ax1.axhline(best_rl["reward"], color="#3FB950", lw=1, linestyle="--", alpha=0.7, label=f"Best={best_rl['reward']:.4f}")
    ax1.set_title("Q-Learning — Reward per Episodio", color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Episodio", color="#8B949E"); ax1.set_ylabel("Reward", color="#8B949E")
    ax1.tick_params(colors="#8B949E"); ax1.spines[:].set_color("#21262D")
    ax1.legend(fontsize=9, facecolor="#161B22", labelcolor="white")
    ax1.grid(True, alpha=0.2, color="#30363D")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#161B22")
    ax2.plot(history["episode"], hist_e, color="#3FB950", lw=2)
    ax2.fill_between(history["episode"], hist_e, alpha=0.15, color="#3FB950")
    ax2.set_title("Epsilon Decay", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Episodio", color="#8B949E"); ax2.set_ylabel("ε", color="#8B949E")
    ax2.tick_params(colors="#8B949E"); ax2.spines[:].set_color("#21262D")
    ax2.grid(True, alpha=0.2, color="#30363D")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161B22")
    ax3r = ax3.twinx(); ax3r.set_facecolor("#161B22")
    ax3.plot(history["episode"], hist_b*100, color="#F8C53A", lw=2)
    ax3r.plot(history["episode"], hist_q, color="#D2A8FF", lw=1.5, linestyle="--")
    ax3.set_title("BEE scoperta & Q-max", color="white", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Episodio", color="#8B949E")
    ax3.set_ylabel("BEE (%)", color="#F8C53A")
    ax3r.set_ylabel("Q-max", color="#D2A8FF")
    ax3.tick_params(colors="#8B949E"); ax3r.tick_params(colors="#D2A8FF")
    ax3.spines[:].set_color("#21262D"); ax3r.spines[:].set_color("#21262D")
    ax3.grid(True, alpha=0.2, color="#30363D")

    fig.suptitle("ARES RL Optimizer — Q-Learning Convergence", color="white",
                 fontsize=13, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Top-5 Configurazioni Scoperte**")
    t5 = top5_rl[["rank","route","coating","fleet","bee","reward","q_value"]].copy()
    t5["bee"]  = (t5["bee"] * 100).round(1).astype(str) + "%"
    t5.columns = ["#","Route","Coating","Fleet","BEE","Reward","Q-value"]
    st.dataframe(t5, use_container_width=True, height=220)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#21262D;margin:30px 0 10px 0'>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#484F58;font-size:0.75rem;text-align:center'>
⚠️ ARES-Sentinel-GBM v2.1 — Simulazione computazionale a scopo di ricerca.
Non costituisce raccomandazione clinica. Validazione richiesta: in vitro → organoidi → in vivo → trial clinici.
&nbsp;|&nbsp; Dati: TCGA-GBM (Brennan et al. Cell 2013)
</p>
""", unsafe_allow_html=True)
