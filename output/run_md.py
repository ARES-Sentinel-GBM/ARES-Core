"""
run_md.py
---------
Modulo 3 della pipeline GBM: Dinamica Molecolare + MM-GBSA.

Simula i top complessi dal docking e valida la stabilità dinamica.
Produce:
  - Traiettorie MD per ogni complesso
  - ΔG_binding MM-GBSA
  - Score di stabilità strutturale
  - Ranking finale integrato (docking + MD)

Esegui:
    python run_md.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path

from modules.md_engine import MDSimulation, MDConfig, MMGBSACalculator, analyze_trajectory
from modules.md_viz import plot_md_trajectories, plot_mmgbsa_comparison, plot_3d_pocket_snapshot


# ─────────────────────────────────────────────────────────────────────
# Top complessi dal Modulo 2 (docking)
# Gene, drug, pocket — ordinati per overall_score
# ─────────────────────────────────────────────────────────────────────
TOP_COMPLEXES = [
    {"gene": "MDM2",   "drug": "milademetan", "pocket": "p53_binding_cleft"},
    {"gene": "PIK3CA", "drug": "BKM120",      "pocket": "ATP_site"},
    {"gene": "CDK4",   "drug": "palbociclib", "pocket": "ATP_site"},
    {"gene": "EGFR",   "drug": "erlotinib",   "pocket": "ATP_site"},
]


def sep(title=""):
    w = 68
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * w)


def run_md_pipeline():
    print("\n" + "═"*68)
    print("  GBM PIPELINE — MODULO 3: DINAMICA MOLECOLARE + MM-GBSA")
    print("  Velocity Verlet + Langevin NVT | MM-GBSA endpoint method")
    print("  T = 310 K (fisiologica) | Campo di forze CG calibrato")
    print("═"*68)

    # ── Config MD ──
    cfg = MDConfig(
        dt_ps          = 0.002,
        n_steps_equil  = 600,
        n_steps_prod   = 6000,
        save_every     = 60,
        temp_K         = 310.0,
        gamma_ps       = 1.0,
        restraint_prot = 2.0,
        verbose        = True,
    )

    mmgbsa_calc = MMGBSACalculator()
    all_results = []

    sep("SIMULAZIONI MD")

    for cx in TOP_COMPLEXES:
        gene   = cx["gene"]
        drug   = cx["drug"]
        pocket = cx["pocket"]

        t0 = time.time()

        # Crea e lancia simulazione
        sim  = MDSimulation(gene, drug, pocket, config=cfg)
        traj = sim.run()

        # Analisi traiettoria
        analysis = analyze_trajectory(traj)

        # MM-GBSA
        mmgbsa = mmgbsa_calc.calculate(traj, sim.ff)

        elapsed = time.time() - t0
        print(f"  ✓ {drug} → {gene}: "
              f"ΔG={mmgbsa['dG_total']:+.1f} kcal/mol  "
              f"Kd≈{mmgbsa['Kd_predicted_nM']:.1f} nM  "
              f"stab={analysis['stability_score']:.3f}  "
              f"[{elapsed:.1f}s]")

        all_results.append({
            "gene":     gene,
            "drug":     drug,
            "pocket":   pocket,
            "traj":     traj,
            "analysis": analysis,
            "mmgbsa":   mmgbsa,
            "sim":      sim,
        })

    sep("ANALISI TRAIETTORIE")

    for r in all_results:
        an = r["analysis"]
        mm = r["mmgbsa"]
        print(f"\n  {r['drug']} → {r['gene']} [{r['pocket'][:20]}]")
        print(f"    Temperatura media:   {an['T_mean_K']:.1f} K  (target: 310 K)")
        print(f"    RMSD proteina:       {an['rmsd_prot_mean']:.3f} Å")
        print(f"    RMSD ligando:        {an['rmsd_lig_mean']:.3f} Å  "
              f"({'stabile' if an['ligand_stable'] else '⚠ fluttuante'})")
        print(f"    RMSF ligando:        {an['rmsf_ligand']:.3f} Å")
        print(f"    H-bond medio:        {an['hbond_mean']:.1f}  (max={an['hbond_max']})")
        print(f"    Stabilità score:     {an['stability_score']:.4f}")
        print(f"    MM-GBSA ΔG:          {mm['dG_total']:+.2f} ± {mm['dG_sem']:.2f} kcal/mol")
        print(f"      ΔE_MM={mm['dG_MM']:+.1f}  ΔG_GB={mm['dG_GB']:+.1f}  "
              f"ΔG_SA={mm['dG_SA']:+.1f}  -TΔS={-mm['T_dS']:+.1f}")
        print(f"    Kd (MM-GBSA):        {mm['Kd_predicted_nM']:.2f} nM")

    sep("RANKING INTEGRATO FINALE")

    # Score composito: docking + MD stabilità + ΔG MM-GBSA
    ranking = []
    for r in all_results:
        dg_norm    = max(0, min(1, (-r["mmgbsa"]["dG_total"] - 2) / 12))
        stab_norm  = r["analysis"]["stability_score"]
        hb_norm    = min(r["analysis"]["hbond_mean"] / 3.0, 1.0)
        final_score = 0.50 * dg_norm + 0.35 * stab_norm + 0.15 * hb_norm

        ranking.append({
            "drug":          r["drug"],
            "gene":          r["gene"],
            "pocket":        r["pocket"],
            "dG_mmgbsa":     r["mmgbsa"]["dG_total"],
            "Kd_nM":         r["mmgbsa"]["Kd_predicted_nM"],
            "stability":     r["analysis"]["stability_score"],
            "hbond_mean":    r["analysis"]["hbond_mean"],
            "rmsd_lig":      r["analysis"]["rmsd_lig_mean"],
            "final_score":   round(final_score, 4),
        })

    ranking.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"\n{'Rank':<5} {'Drug':<14} {'Gene':<8} {'ΔG (kcal/mol)':<16} "
          f"{'Kd (nM)':<10} {'Stab.':<8} {'H-bond':<8} {'Final'}")
    print("─" * 78)
    for i, r in enumerate(ranking, 1):
        print(f"  [{i}]  {r['drug']:<12} {r['gene']:<8} "
              f"{r['dG_mmgbsa']:>+8.2f}       "
              f"{r['Kd_nM']:>8.2f}   "
              f"{r['stability']:.3f}   "
              f"{r['hbond_mean']:.1f}     "
              f"{r['final_score']:.4f}")

    sep("VISUALIZZAZIONI")

    p8 = plot_md_trajectories(all_results)
    print(f"  ✓ {p8.name}")

    p9 = plot_mmgbsa_comparison(all_results)
    print(f"  ✓ {p9.name}")

    # Snapshot 3D del candidato #1
    best = all_results[0]
    p10 = plot_3d_pocket_snapshot(best["traj"], idx_frame=-1)
    if p10:
        print(f"  ✓ {p10.name}")

    sep("CONCLUSIONI")

    best_r = ranking[0]
    print(f"\nCandidato computazionale finale #1:")
    print(f"  {best_r['drug']} → {best_r['gene']} ({best_r['pocket']})")
    print(f"  ΔG_binding:  {best_r['dG_mmgbsa']:+.2f} kcal/mol")
    print(f"  Kd predetto: {best_r['Kd_nM']:.2f} nM")
    print(f"  Stabilità:   {best_r['stability']:.3f} / 1.0")
    print(f"  Score finale:{best_r['final_score']:.4f}")

    print("\nProssimi step scientifici:")
    next_steps = [
        "AutoDock Vina: docking ad alta risoluzione su strutture PDB reali",
        "GROMACS/AMBER: MD all-atom (100-500 ns) per validazione completa",
        "MM-PBSA/FEP: calcolo ΔΔG per mutazioni di resistenza",
        "Test in vitro: IC50 su linee GBM (U87MG, T98G, GSC11)",
        "Profilo ADMET: tossicità CNS, metabolismo epatico (CYP450)",
        "Ottimizzazione farmaco: SAR guidata dai dati MD",
    ]
    for i, s in enumerate(next_steps, 1):
        print(f"  {i}. {s}")

    sep()
    print("⚠  Simulazione CG semplificata. Per pubblicazione scientifica")
    print("   richiede validazione con MD all-atom (GROMACS/AMBER).\n")

    # Salva tabella ranking
    df_rank = pd.DataFrame(ranking)
    df_rank.to_csv(Path(__file__).parent / "output" / "md_ranking.csv", index=False)
    print("  ✓ md_ranking.csv")

    return ranking


if __name__ == "__main__":
    run_md_pipeline()
