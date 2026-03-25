"""
run_admet.py — Modulo 4: ADMET + Farmacocinetica CNS

Pipeline completa: per ogni candidato GBM calcola
  A) Assorbimento orale e solubilità
  B) Distribuzione: Vd, PPB, Kp,uu, BEE multi-meccanismo
  C) Metabolismo: CYP450, clearance, t½
  D) Escrezione: renale vs epatica
  E) Tossicità: hERG, DILI, neurotox, AMES
  F) Delivery optimizer: rotta ottimale per ogni candidato
  G) Ranking integrato finale: Docking + MD + ADMET
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path

from modules.admet_profiler import (
    AbsorptionModel, DistributionModel, MetabolismModel,
    ToxicityModel, DeliveryOptimizer, DRUG_EXTENDED, CYP450_PROFILES
)
from modules.admet_viz import (
    plot_admet_radar, plot_bee_strategies, plot_toxicity_dashboard,
    plot_cyp450_matrix, plot_final_integrated_ranking
)


# ─────────────────────────────────────────────────────────────────────
# Candidati (top 4 da Modulo 3 MD) + comparatore SOC (temozolomide)
# ─────────────────────────────────────────────────────────────────────
CANDIDATES = [
    {"drug": "milademetan", "gene": "MDM2",   "docking_score": 0.9022, "md_score": 0.6197},
    {"drug": "BKM120",      "gene": "PIK3CA", "docking_score": 0.7723, "md_score": 0.8628},
    {"drug": "palbociclib", "gene": "CDK4",   "docking_score": 0.7441, "md_score": 0.8333},
    {"drug": "erlotinib",   "gene": "EGFR",   "docking_score": 0.6688, "md_score": 0.8025},
    {"drug": "temozolomide","gene": "DNA",     "docking_score": 0.5000, "md_score": 0.5000},  # SOC reference
]


def sep(title=""):
    w = 68
    pad = (w - len(title) - 2) // 2 if title else 0
    print(f"\n{'─'*pad} {title} {'─'*pad}" if title else "─"*w)


def run_admet_pipeline():
    print("\n" + "═"*68)
    print("  GBM PIPELINE — MODULO 4: ADMET + FARMACOCINETICA CNS")
    print("  BIOS skill: BEE multi-meccanismo | P-gp | CYP450 | hERG")
    print("  T=310K | pH=7.4 | 70 kg paziente adulto")
    print("═"*68)

    abs_model  = AbsorptionModel()
    dist_model = DistributionModel()
    metab_model = MetabolismModel()
    tox_model  = ToxicityModel()
    deliv_opt  = DeliveryOptimizer()

    admet_rows  = []
    bee_rows    = []
    cyp_rows    = []

    sep("PROFILI ADMET COMPLETI")

    for cx in CANDIDATES:
        drug = cx["drug"]
        d    = DRUG_EXTENDED.get(drug, {})
        print(f"\n  ━━ {drug.upper()} → {cx['gene']} ━━")
        print(f"     Meccanismo: {d.get('mechanism','N/D')}")
        print(f"     Fase clinica: {d.get('clinical_phase','N/D')}")

        # A — Assorbimento
        ab = abs_model.predict(drug)
        print(f"\n  [A] ASSORBIMENTO")
        print(f"     F% orale:    {ab.get('F_oral_final',0)*100:.0f}%  ({ab.get('F_source','?')})")
        print(f"     Papp (PAMPA):{ab.get('log_papp_cms',0):.2f} log cm/s  "
              f"[{ab.get('absorption_class','?')}]")
        print(f"     Solubilità:  {ab.get('logS_aqueous',0):.2f} logS  "
              f"({ab.get('solubility_mgmL',0):.4f} mg/mL)")

        # D — Distribuzione
        dist = dist_model.predict(drug, delivery_route="oral_passive")
        print(f"\n  [D] DISTRIBUZIONE")
        print(f"     Vd:          {dist.get('Vd_Lkg',0):.2f} L/kg")
        print(f"     PPB:         {dist.get('PPB_fraction',0)*100:.0f}%  "
              f"(fu={dist.get('fu_plasma',0):.3f})")
        print(f"     CNS MPO:     {dist.get('CNS_MPO',0):.2f} / 6.0")
        print(f"     P-gp:        {'Substrato ⚠' if dist.get('Pgp_substrate') else 'Non substrato ✓'}")
        print(f"     Kp,uu:       {dist.get('Kpuu',0):.4f}  "
              f"({'✓ adeguato' if dist.get('Kpuu_adequate') else '✗ insufficiente'})")
        print(f"     fu,brain:    {dist.get('fu_brain',0):.4f}")
        print(f"     C_GBM rel.:  {dist.get('C_gbm_relative',0):.5f}")

        # M — Metabolismo
        met = metab_model.predict(drug, dist.get("Vd_Lkg",1), dist.get("fu_plasma",0.1))
        print(f"\n  [M] METABOLISMO")
        print(f"     t½:          {met.get('t_half_final_h',0):.1f} h  ({met.get('t_half_source','?')})")
        print(f"     Cl_epatica:  {met.get('Cl_hepatic_mLmin',0):.0f} mL/min  "
              f"(ER={met.get('ER_hepatic',0):.2f}, {met.get('extraction_class','?')})")
        print(f"     DDI risk:    {met.get('DDI_risk','?')}")
        if met.get("CYP_profile"):
            for iso, flag in met["CYP_profile"].items():
                print(f"     CYP{iso}:    {flag}")

        # E — Escrezione (dalla clearance totale)
        print(f"\n  [E] ESCREZIONE")
        print(f"     Cl_renale:   {met.get('Cl_renal_mLmin',0):.0f} mL/min")
        print(f"     Cl_totale:   {met.get('Cl_total_mLmin',0):.0f} mL/min")

        # T — Tossicità
        tox = tox_model.predict(drug, dist.get("CNS_MPO",4), met.get("n_CYP_inhibitors",0))
        print(f"\n  [T] TOSSICITÀ")
        print(f"     hERG:        {tox.get('hERG_risk','?')}  (score={tox.get('hERG_score',0):.2f})")
        print(f"     DILI:        {tox.get('DILI_risk','?')}  (score={tox.get('DILI_score',0):.2f})")
        print(f"     Neurotox:    {tox.get('neuro_risk','?')}")
        print(f"     AMES:        {tox.get('AMES','?')}")
        print(f"     Safety:      {tox.get('safety_class','?')}")

        # F — Delivery optimization per tutte le rotte
        deliv = deliv_opt.optimize(drug, dist, met)
        print(f"\n  [F] DELIVERY OTTIMALE")
        print(f"     Rotta:       {deliv.get('optimal_route','?').replace('_',' ')}  "
              f"(BEE={deliv.get('optimal_score',0)*100:.0f}%)")
        print(f"     {deliv.get('recommendation','')}")

        # Calcola BEE per tutte le rotte (per heatmap)
        for route in ["oral_passive","RMT_transferrin","FUS","intranasale","exosome"]:
            b = dist_model._bee_by_route(
                route,
                d.get("logD74",2), d.get("MW",400), d.get("TPSA",80),
                d.get("HBD",2), dist.get("CNS_MPO",4),
                d.get("pgp_substrate",False), dist.get("Kpuu",0.2)
            )
            bee_rows.append({"drug": drug, "route": route, "bee_fraction": b["fraction"]})

        # CYP profilo per matrice
        for iso, data in CYP450_PROFILES.get(drug, {}).items():
            cyp_rows.append({
                "drug": drug, "isoform": iso,
                "substrate": data.get("substrate", False),
                "inhibitor": data.get("inhibitor", False),
                "strength":  data.get("strength", ""),
            })

        # Score ADMET composito (0-1)
        s_absorption = ab.get("F_oral_final", 0.5)
        s_bee        = deliv.get("optimal_score", 0.3)
        s_kpuu       = min(dist.get("Kpuu",0.1) * 2, 1.0)
        s_metab      = 1 - min(met.get("ER_hepatic",0.5), 1.0)
        s_safety     = 1 - tox.get("overall_tox_score", 0.3)
        s_mpo        = dist.get("CNS_MPO",4) / 6.0
        admet_score  = (0.20*s_absorption + 0.25*s_bee + 0.20*s_kpuu +
                        0.15*s_metab + 0.20*s_safety)

        admet_rows.append({
            "drug":              drug,
            "gene":              cx["gene"],
            "F_oral_final":      ab.get("F_oral_final",0),
            "logS":              ab.get("logS_aqueous",0),
            "Vd_Lkg":            dist.get("Vd_Lkg",0),
            "PPB":               dist.get("PPB_fraction",0),
            "Kpuu":              dist.get("Kpuu",0),
            "CNS_MPO":           dist.get("CNS_MPO",0),
            "Pgp_substrate":     dist.get("Pgp_substrate",False),
            "BEE_optimal_frac":  deliv.get("optimal_score",0),
            "BEE_optimal_pct":   deliv.get("optimal_score",0)*100,
            "optimal_route":     deliv.get("optimal_route",""),
            "t_half_h":          met.get("t_half_final_h",0),
            "ER_hepatic":        met.get("ER_hepatic",0),
            "DDI_risk":          met.get("DDI_risk",""),
            "hERG_risk":         tox.get("hERG_risk",""),
            "DILI_risk":         tox.get("DILI_risk",""),
            "neuro_risk":        tox.get("neuro_risk",""),
            "AMES":              tox.get("AMES",""),
            "overall_tox_score": tox.get("overall_tox_score",0),
            "safety_class":      tox.get("safety_class",""),
            "admet_score":       round(admet_score, 4),
            "docking_score":     cx["docking_score"],
            "md_score":          cx["md_score"],
        })

    # ─────────────────────────────────────────────────────────────────
    # RANKING INTEGRATO FINALE
    # ─────────────────────────────────────────────────────────────────
    sep("RANKING INTEGRATO FINALE")

    admet_df = pd.DataFrame(admet_rows)
    bee_df   = pd.DataFrame(bee_rows)
    cyp_df   = pd.DataFrame(cyp_rows)

    # Score integrato: 35% docking + 35% MD + 30% ADMET
    admet_df["score_docking"] = admet_df["docking_score"]
    admet_df["score_md"]      = admet_df["md_score"]
    admet_df["score_admet"]   = admet_df["admet_score"]
    admet_df["integrated_score"] = (
        0.35 * admet_df["docking_score"] +
        0.35 * admet_df["md_score"]      +
        0.30 * admet_df["admet_score"]
    ).round(4)

    admet_df = admet_df.sort_values("integrated_score", ascending=False).reset_index(drop=True)

    print(f"\n{'Rank':<5} {'Drug':<14} {'Gene':<8} {'Docking':<9} {'MD':<9} "
          f"{'ADMET':<9} {'Integrated':<11} {'Rotta BEE':<20} {'Safety'}")
    print("─"*90)
    for i, row in admet_df.iterrows():
        rank = i + 1
        print(f"  [{rank}]  {row['drug']:<12} {row['gene']:<8} "
              f"{row['docking_score']:.3f}    {row['md_score']:.3f}    "
              f"{row['admet_score']:.3f}    {row['integrated_score']:.4f}      "
              f"{row['optimal_route'][:16]:<20} {row['safety_class']}")

    sep("VISUALIZZAZIONI")

    p11 = plot_admet_radar(admet_df)
    print(f"  ✓ {p11.name}")

    p12 = plot_bee_strategies(bee_df)
    print(f"  ✓ {p12.name}")

    p13 = plot_toxicity_dashboard(admet_df)
    print(f"  ✓ {p13.name}")

    p14 = plot_cyp450_matrix(cyp_df)
    print(f"  ✓ {p14.name}")

    p15 = plot_final_integrated_ranking(admet_df)
    print(f"  ✓ {p15.name}")

    # Salva CSV finale
    out_csv = Path(__file__).parent / "output" / "final_pipeline_ranking.csv"
    admet_df.to_csv(out_csv, index=False)
    print(f"  ✓ final_pipeline_ranking.csv")

    sep("CANDIDATO FINALE PIPELINE COMPLETA")

    winner = admet_df.iloc[0]
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  CANDIDATO #1 — Pipeline GBM Computazionale         │
  │                                                     │
  │  Farmaco:   {winner['drug']:<40} │
  │  Target:    {winner['gene']:<40} │
  │  ─────────────────────────────────────────────────  │
  │  Score integrato:  {winner['integrated_score']:.4f}                          │
  │  Score docking:    {winner['docking_score']:.4f} (AlphaFold + empirico)     │
  │  Score MD/GBSA:    {winner['md_score']:.4f} (CG Langevin NVT)          │
  │  Score ADMET:      {winner['admet_score']:.4f} (BIOS skill)               │
  │  ─────────────────────────────────────────────────  │
  │  Rotta delivery:   {winner['optimal_route']:<35} │
  │  BEE penetraz.:    {winner['BEE_optimal_pct']:.0f}%                               │
  │  Kp,uu:            {winner['Kpuu']:.4f}                                 │
  │  t½:               {winner['t_half_h']:.1f} h                                │
  │  Safety:           {winner['safety_class']:<35} │
  └─────────────────────────────────────────────────────┘
    """)

    print("Validazione sperimentale consigliata:")
    steps = [
        f"IC50 in vitro su linee GBM: U87MG, T98G, GSC11 (target {winner['gene']})",
        f"Saggi BEE: hCMEC/D3 monostrato + Transwell per confermare {winner['BEE_optimal_pct']:.0f}% penetrazione",
        "Profilo metabolico completo: microsomi epatici umani + epatociti in sospensione",
        "hERG patch-clamp: conferma sicurezza cardiaca",
        "Modello ortotopico murino GBM (U87-luc intracranico): efficacia + tossicità in vivo",
        "PK brain/plasma: rapporto misurato per calibrare modello Kp,uu",
    ]
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    sep()
    print("⚠  Pipeline computazionale completa (4 moduli).")
    print("   Tutti i valori richiedono validazione sperimentale.\n")

    return admet_df


if __name__ == "__main__":
    run_admet_pipeline()
