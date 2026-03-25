"""
run_transformation.py — TransformationEngine integrato nella pipeline GBM
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from modules.transformation_engine import (
    TransformationEngine, MultiOncometaboliteScorer, ONCOMETABOLITES,
    DRUG_ONCOMETABOLITE_ACTIVITY
)

OUTPUT = Path(__file__).parent / "output"

# Dati integrati dalla pipeline ADMET (Modulo 4)
PIPELINE_DRUG_DATA = [
    {"drug":"milademetan","gene":"MDM2",   "Kpuu":0.0575,"t_half_h":20.0,"integrated_score":0.6951,"admet_score":0.5410},
    {"drug":"BKM120",     "gene":"PIK3CA", "Kpuu":0.0933,"t_half_h":30.0,"integrated_score":0.7675,"admet_score":0.6506},
    {"drug":"palbociclib","gene":"CDK4",   "Kpuu":0.1248,"t_half_h":26.0,"integrated_score":0.7500,"admet_score":0.6599},
    {"drug":"erlotinib",  "gene":"EGFR",   "Kpuu":0.0446,"t_half_h":36.0,"integrated_score":0.7050,"admet_score":0.6326},
    {"drug":"ivosidenib", "gene":"IDH1",   "Kpuu":0.20,  "t_half_h":93.0,"integrated_score":0.0,   "admet_score":0.0},  # gold std 2-HG
    {"drug":"temozolomide","gene":"DNA",   "Kpuu":1.3797,"t_half_h":1.8, "integrated_score":0.6257,"admet_score":0.9190},
]

P = {"bg":"#0d1520","panel":"#111a28","grid":"#1e3050","text":"#c8d8e8",
     "cyan":"#00f5d4","red":"#ff006e","yellow":"#ffd60a","green":"#06d6a0",
     "blue":"#00b4d8","purple":"#9b5de5","orange":"#f4a261","neutral":"#6c757d"}

def sep(t=""):
    w=68; pad=(w-len(t)-2)//2 if t else 0
    print(f"\n{'─'*pad} {t} {'─'*pad}" if t else "─"*w)


def plot_transformation_results(profiles: list, multi_scores: list) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=P["bg"])
    fig.suptitle("TRANSFORMATION ENGINE — FEGATO SINTETICO MOLECOLARE | GBM",
                 color=P["cyan"], fontsize=12, fontweight="bold")

    def ax_dark(ax, t="", xl="", yl=""):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=7.5)
        for sp in ["bottom","left"]: ax.spines[sp].set_color(P["grid"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color(P["text"]); ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["cyan"])
        ax.grid(True, color=P["grid"], lw=0.5, alpha=0.5)
        if t: ax.set_title(t, fontsize=9, fontweight="bold", pad=7)
        if xl: ax.set_xlabel(xl, fontsize=8)
        if yl: ax.set_ylabel(yl, fontsize=8)

    drugs = [p["drug"] for p in profiles]
    colors = [P["cyan"], P["red"], P["yellow"], P["green"], P["orange"], P["blue"]]

    # 1. Michaelis-Menten curve per D-2HG
    ax1 = axes[0, 0]
    conc_range = np.linspace(0, 30, 200)
    engine_2hg = TransformationEngine("D-2HG")
    for i, p in enumerate(profiles[:4]):
        drug = p["drug"]
        eff  = DRUG_ONCOMETABOLITE_ACTIVITY.get(drug,{}).get("D-2HG",{}).get("direct_activity", 0.03)
        eff  = max(eff, 0.03)
        v    = [(engine_2hg.kcat_target * c / (engine_2hg.km_affinity + c)) * eff
                for c in conc_range]
        ax1.plot(conc_range, v, color=colors[i], lw=2, label=drug, alpha=0.85)
    ax1.axvline(x=15, color=P["red"],    ls="--", lw=0.8, alpha=0.6)
    ax1.axvline(x=0.5, color=P["yellow"],ls="--", lw=0.8, alpha=0.6)
    ax1.text(15.2, 2, "[2-HG]\ntumor", color=P["red"],    fontsize=6.5)
    ax1.text(0.6,  2, "Km",            color=P["yellow"], fontsize=6.5)
    ax1.legend(fontsize=7, facecolor=P["bg"], edgecolor=P["grid"], labelcolor=P["text"])
    ax_dark(ax1, "Cinetica MM — D-2HG (IDH1 R132H)", "[2-HG] mM", "V (rxn/s)")

    # 2. Neutralization score per oncometabolita
    ax2 = axes[0, 1]
    oms  = ["D-2HG", "L-Lactate", "Fumarate", "Succinate"]
    x    = np.arange(len(drugs[:5]))
    bar_w = 0.18
    om_colors = [P["red"], P["yellow"], P["orange"], P["purple"]]
    for j, om in enumerate(oms):
        vals = [ms["per_oncometabolite"][om]["score"] for ms in multi_scores[:5]]
        ax2.bar(x + j*bar_w - 1.5*bar_w, vals, bar_w, label=om,
                color=om_colors[j], alpha=0.85, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(drugs[:5], rotation=25, ha="right", fontsize=7.5, color=P["text"])
    ax2.legend(fontsize=7, facecolor=P["bg"], edgecolor=P["grid"], labelcolor=P["text"])
    ax_dark(ax2, "Score per Oncometabolita", "", "MN Score (0-1)")

    # 3. Score aggregato + ranking
    ax3 = axes[0, 2]
    agg  = [ms["aggregate_score"] for ms in multi_scores]
    cols = [P["red"] if v > 0.4 else P["yellow"] if v > 0.2 else P["neutral"] for v in agg]
    bars = ax3.barh(drugs, agg, color=cols, height=0.7, edgecolor="none", alpha=0.9)
    for b, v in zip(bars, agg):
        ax3.text(v+0.005, b.get_y()+b.get_height()/2, f"{v:.4f}",
                 va="center", fontsize=7.5, color=P["text"])
    ax3.axvline(x=0.40, color=P["red"],    ls="--", lw=0.8, alpha=0.5)
    ax3.axvline(x=0.20, color=P["yellow"], ls="--", lw=0.8, alpha=0.5)
    ax3.invert_yaxis()
    ax_dark(ax3, "Score Aggregato Fegato Sintetico", "Aggregate Score", "")

    # 4. Riduzione oncometabolita in 24h
    ax4 = axes[1, 0]
    for i, p in enumerate(profiles[:5]):
        ax4.scatter(p["kpuu"], p["conc_reduction_24h_pct"],
                    s=p["t_half_h"]*4, c=colors[i], alpha=0.9,
                    edgecolors="white", linewidths=0.8, zorder=3)
        ax4.annotate(p["drug"][:6], (p["kpuu"], p["conc_reduction_24h_pct"]),
                     fontsize=7, color=P["text"], xytext=(4,3), textcoords="offset points")
    ax4.axhline(y=50, color=P["yellow"], ls="--", lw=0.8, alpha=0.5)
    ax4.text(0.01, 51, "50% riduzione",  color=P["yellow"], fontsize=7)
    ax4.set_xlabel("Kp,uu (brain/plasma)", color=P["text"], fontsize=8)
    ax4.set_ylabel("Riduzione [2-HG] 24h (%)", color=P["text"], fontsize=8)
    ax4.set_title("Riduzione Oncometabolita in 24h\n(dimensione bolla = t½)", color=P["cyan"],
                  fontsize=9, fontweight="bold")
    ax4.set_facecolor(P["panel"])
    ax4.tick_params(colors=P["text"])
    for sp in ["bottom","left"]: ax4.spines[sp].set_color(P["grid"])
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    ax4.grid(True, color=P["grid"], lw=0.5, alpha=0.5)

    # 5. Ranking integrato FINALE (pipeline + transformation)
    ax5 = axes[1, 1]
    final_scores = []
    for dd, ms in zip(PIPELINE_DRUG_DATA, multi_scores):
        pip_score = dd.get("integrated_score", 0)
        tr_score  = ms["aggregate_score"]
        # Score finale: 70% pipeline + 30% transformation engine
        final = 0.70 * pip_score + 0.30 * tr_score if pip_score > 0 else tr_score
        final_scores.append({"drug": dd["drug"], "final": round(final,4),
                              "pipeline": pip_score, "transform": tr_score})
    final_scores.sort(key=lambda x: x["final"], reverse=True)

    y_pos = range(len(final_scores))
    ax5.barh(y_pos, [f["final"] for f in final_scores],
             color=[P["cyan"] if f["final"]>0.60 else P["yellow"] for f in final_scores],
             height=0.7, edgecolor="none", alpha=0.9)
    for i, f in enumerate(final_scores):
        ax5.text(f["final"]+0.005, i, f"{f['final']:.4f}", va="center",
                 fontsize=8, color=P["text"])
    ax5.set_yticks(list(y_pos))
    ax5.set_yticklabels([f["drug"] for f in final_scores], fontsize=9, color=P["text"])
    ax5.invert_yaxis()
    ax_dark(ax5, "RANKING FINALE (Pipeline + Fegato Sintetico)", "Score (0-1)", "")

    # 6. Pannello biologico
    ax6 = axes[1, 2]
    ax6.axis("off"); ax6.set_facecolor(P["panel"])
    ax6.set_title("BIOLOGIA — ONCOMETABOLITI GBM", color=P["cyan"],
                  fontsize=9.5, fontweight="bold")
    bio_lines = [
        ("D-2HG",      f"IDH1 R132H → {ONCOMETABOLITES['D-2HG']['gbm_prevalence']*100:.0f}% GBM | {ONCOMETABOLITES['D-2HG']['conc_tumor_mM']} mM"),
        ("L-Lattato",  f"Effetto Warburg → {ONCOMETABOLITES['L-Lactate']['gbm_prevalence']*100:.0f}% GBM | {ONCOMETABOLITES['L-Lactate']['conc_tumor_mM']} mM"),
        ("Fumarato",   f"FH loss → {ONCOMETABOLITES['Fumarate']['gbm_prevalence']*100:.1f}% GBM | {ONCOMETABOLITES['Fumarate']['conc_tumor_mM']} mM"),
        ("Succinato",  f"SDH loss → {ONCOMETABOLITES['Succinate']['gbm_prevalence']*100:.1f}% GBM | {ONCOMETABOLITES['Succinate']['conc_tumor_mM']} mM"),
        ("","",""),
        ("FORMULA",    "V = Vmax·[S] / (Km + [S])"),
        ("Vmax",       f"kcat · [E] = {ONCOMETABOLITES['D-2HG']['kcat_s']} rxn/s"),
        ("Km (D-2HG)", f"{ONCOMETABOLITES['D-2HG']['Km_mM']} mM (affinità)"),
        ("kcat/Km",    f"{ONCOMETABOLITES['D-2HG']['kcat_s']/ONCOMETABOLITES['D-2HG']['Km_mM']:.0f} mM⁻¹s⁻¹"),
    ]
    for i, line in enumerate(bio_lines):
        y = 0.94 - i*0.09
        if len(line) >= 2 and line[0]:
            ax6.text(0.02, y, line[0]+":", transform=ax6.transAxes,
                     color=P["cyan"], fontsize=7.5, va="top", fontweight="bold")
            ax6.text(0.30, y, line[1] if len(line) > 1 else "",
                     transform=ax6.transAxes, color=P["text"], fontsize=7, va="top")

    plt.tight_layout()
    path = OUTPUT / "17_transformation_engine.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    return path


def run_transformation_pipeline():
    print("\n" + "═"*68)
    print("  GBM PIPELINE — TRANSFORMATION ENGINE")
    print("  'Fegato Sintetico Molecolare' — Michaelis-Menten")
    print("  Oncometaboliti: D-2HG | L-Lattato | Fumarato | Succinato")
    print("═"*68)

    sep("PROFILI ONCOMETABOLITI BERSAGLIO")
    for name, om in ONCOMETABOLITES.items():
        if name == "N-Acetylaspartate": continue
        print(f"\n  {name}")
        print(f"    Origine:      {om['gene_origin']}")
        print(f"    Prevalenza:   {om['gbm_prevalence']*100:.1f}% GBM")
        print(f"    [tumore]:     {om['conc_tumor_mM']} mM  vs  normale {om['conc_normal_mM']} mM")
        print(f"    kcat:         {om['kcat_s']} s⁻¹   Km: {om['Km_mM']} mM   kcat/Km: {om['kcat_s']/om['Km_mM']:.0f} mM⁻¹s⁻¹")
        print(f"    Farmaco ref:  {om['drug_reference']}")

    sep("ANALISI FEGATO SINTETICO PER CANDIDATO")

    engine_2hg = TransformationEngine("D-2HG")
    multi_scorer = MultiOncometaboliteScorer()
    profiles_2hg = []
    multi_scores  = []

    for drug_data in PIPELINE_DRUG_DATA:
        drug = drug_data["drug"]
        print(f"\n  ━━ {drug.upper()} ━━")

        profile = engine_2hg.full_profile(drug_data)
        profiles_2hg.append(profile)

        multi = multi_scorer.score_all(drug_data)
        multi_scores.append(multi)

        print(f"  Meccanismo:          {profile['indirect_mechanism']}")
        print(f"  [2-HG] effettiva CNS:{profile['conc_cns_effective_mM']:.4f} mM  "
              f"(tumore: {profile['conc_tumor_mM']} mM × Kpuu={profile['kpuu']:.4f})")
        print(f"  Efficienza enzimatica:{profile['enzyme_efficiency']:.3f}")
        print(f"  Velocità conversione: {profile['conversion_rate_s']:.4f} rxn/s")
        print(f"  Saturazione Vmax:    {profile['saturation_fraction']*100:.1f}%")
        print(f"  Riduzione [2-HG] 24h:{profile['conc_reduction_24h_pct']:.1f}%")
        print(f"  MN Score (D-2HG):    {profile['metabolic_neutralization_score']:.4f}")
        print(f"  → {profile['interpretation']}")
        print(f"\n  Multi-oncometaboliti:")
        for om, data in multi["per_oncometabolite"].items():
            print(f"    {om:<15} score={data['score']:.4f}  "
                  f"red.24h={data['reduction_24h']:.1f}%")
        print(f"  Aggregate score:     {multi['aggregate_score']:.4f}")
        print(f"  Best target:         {multi['best_target']}")
        print(f"  Note:                {multi['note']}")

    sep("RANKING INTEGRATO FINALE (con TransformationEngine)")

    final_rows = []
    for dd, ms in zip(PIPELINE_DRUG_DATA, multi_scores):
        pip   = dd.get("integrated_score", 0)
        tr    = ms["aggregate_score"]
        final = 0.70*pip + 0.30*tr if pip > 0 else tr
        final_rows.append({
            "drug":             dd["drug"],
            "gene":             dd["gene"],
            "pipeline_score":   pip,
            "transform_score":  tr,
            "final_score":      round(final, 4),
            "best_om_target":   ms["best_target"],
        })
    final_rows.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"\n{'Rank':<5} {'Drug':<14} {'Gene':<8} {'Pipeline':<10} "
          f"{'Transform':<11} {'Final':<8} {'Best OM target'}")
    print("─"*70)
    for i, r in enumerate(final_rows, 1):
        print(f"  [{i}]  {r['drug']:<12} {r['gene']:<8} "
              f"{r['pipeline_score']:.4f}    {r['transform_score']:.4f}     "
              f"{r['final_score']:.4f}  {r['best_om_target']}")

    sep("VISUALIZZAZIONE")
    p17 = plot_transformation_results(profiles_2hg, multi_scores)
    print(f"  ✓ {p17.name}")

    pd.DataFrame(final_rows).to_csv(OUTPUT / "transformation_ranking.csv", index=False)
    print(f"  ✓ transformation_ranking.csv")

    sep("INSIGHT CHIAVE")
    winner = final_rows[0]
    print(f"""
  Il tuo algoritmo TransformationEngine aggiunge un layer critico:
  valuta non solo 'il farmaco blocca il recettore?'
  ma 'il farmaco neutralizza la tossina che il tumore produce?'

  Candidato #1 con TransformationEngine: {winner['drug']} → {winner['gene']}
  Final score: {winner['final_score']:.4f}

  Osservazione importante:
  → ivosidenib (gold standard anti-D-2HG) non compare nel ranking
    classico perché GBM IDH-wt non ha IDH1 R132H.
    Ma in GBM IDH-mutato (5-8% casi) sarebbe candidato #1 assoluto.
  → Temozolomide ha il Kpuu più alto (1.38) — raggiunge meglio il CNS —
    ma zero attività anti-oncometabolita: classic trade-off.
  → BKM120 via PI3K→mTOR→LDH-A è il più efficace contro L-Lattato (Warburg).
    """)

    sep()
    print("⚠  TransformationEngine integrato come Modulo 6 della pipeline.\n")
    return final_rows


if __name__ == "__main__":
    run_transformation_pipeline()
