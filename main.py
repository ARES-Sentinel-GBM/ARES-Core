"""
main.py — GBM Computational Pipeline
=====================================
Pipeline computazionale per analisi molecolare e simulazione
farmacocinetica in Glioblastoma Multiforme (GBM).

Dati: TCGA-GBM (n=617 pazienti, IDH-wildtype)
Fonte: Brennan CW et al., Cell 2013; Cancer Genome Atlas 2021

Uso:
    python main.py

Output:
    output/01_mutation_landscape.png
    output/02_target_scores.png
    output/03_pk_simulation.png
    output/04_drug_comparison.png
    output/report_summary.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.gbm_data import GBMDataLoader
from modules.target_analyzer import TargetAnalyzer, BEE_PENETRATION
from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute,
    PKParameters, DrugComparison
)
from modules.visualizer import (
    plot_mutation_landscape, plot_target_scores,
    plot_pk_simulation, plot_drug_comparison
)
import pandas as pd
from pathlib import Path


def separator(title=""):
    width = 68
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def run_pipeline():
    print("\n" + "═"*68)
    print("  GBM COMPUTATIONAL PIPELINE v1.0")
    print("  Glioblastoma Multiforme — Analisi Molecolare e Nanodrone PK")
    print("  Dati: TCGA-GBM (n=617) | Brennan et al. Cell 2013")
    print("═"*68)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Carica dati
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 1: CARICAMENTO DATI TCGA-GBM")
    loader = GBMDataLoader(source="embedded")
    # Per dati live: loader = GBMDataLoader(source="api")

    mut_df  = loader.load_mutations()
    expr_df = loader.load_expression()
    drug_df = loader.load_drugs()

    print(f"✓ Mutazioni caricate: {len(mut_df)} geni")
    print(f"✓ Firma espressione:  {len(expr_df)} geni")
    print(f"✓ Database farmaci:   {len(drug_df)} agenti")

    print("\nTop 5 geni più mutati in GBM:")
    for _, row in mut_df.head(5).iterrows():
        print(f"  {row['gene']:<12} {row['frequency']:>5.1f}%  [{row['pathway']}]")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Analisi target
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 2: ANALISI TARGET MOLECOLARI")
    analyzer = TargetAnalyzer(mut_df, expr_df)

    score_df   = analyzer.score_targets()
    pathway_df = analyzer.rank_pathways()
    sl_df      = analyzer.get_synthetic_lethality_map()

    top_targets = score_df[score_df["druggable"] == True].head(5)
    print(f"✓ Target druggabili identificati: {len(score_df[score_df['druggable']==True])}")
    print("\nTop 5 target per composite score:")
    for _, row in top_targets.iterrows():
        print(f"  [{row['priority_rank']:>2}] {row['gene']:<10} "
              f"score={row['composite_score']:.3f}  "
              f"freq={row['mutation_freq']:.1f}%  "
              f"{row['pathway']}")

    print("\nPathway prioritari:")
    for _, row in pathway_df.head(4).iterrows():
        print(f"  {row['pathway']:<20} "
              f"mean_score={row['mean_score']:.3f}  "
              f"druggable_genes={row['druggable_genes']}")

    print(f"\n✓ Coppie letali sintetiche identificate: {len(sl_df)}")
    high_sl = sl_df[sl_df["evidence"] == "HIGH"].head(3)
    for _, row in high_sl.iterrows():
        print(f"  {row['mutated_gene']:>8} mutato → TARGET: {row['target_gene']:<10} "
              f"freq={row['mut_frequency']:.1f}%  [{row['reference']}]")

    # BEE strategy matrix per top target
    top3_genes = list(score_df.head(3)["gene"])
    bee_matrix = analyzer.bee_strategy_matrix(top3_genes)
    print(f"\n✓ BEE Strategy Matrix — Top 3 target: {top3_genes}")
    print(f"  Metodo ottimale: {bee_matrix.iloc[0]['delivery_method']} "
          f"(BEE={bee_matrix.iloc[0]['bee_penetration']*100:.0f}%)")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Simulazione PK Nanodrone
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 3: SIMULAZIONE FARMACOCINETICA NANODRONE")

    # Configura flotta (la tua configurazione: 6S-1D-3A-3L)
    fleet = FleetConfig(
        n_sentinelle=6,
        n_decisori=1,
        n_attacco=3,
        n_lager=3,
        delivery_route=DeliveryRoute.FUS,  # FUS = miglior penetrazione BEE
        drone_spec=DroneSpec(
            size_nm=120,
            surface_coating="PEG+Transferrin",  # RMT via recettore transferrina
            payload_type="chemo",
            immune_evasion=0.72,
            targeting_affinity=0.65,
        )
    )

    pk = PKParameters(
        bee_base=0.05,
        tumor_icp=25.0,  # pressione intracranica GBM elevata
        tumor_ifp=22.0,  # pressione interstiziale tumorale alta
    )

    simulator = NanodronePKSimulator(fleet, pk)
    pk_df = simulator.simulate(dose=1.0, duration_h=72.0, target_gene="EGFR")
    summary = simulator.pk_summary(pk_df)

    print(f"✓ Flotta: {fleet.total_drones} droni totali")
    print(f"  Configurazione: {fleet.n_sentinelle}S / {fleet.n_decisori}D / "
          f"{fleet.n_attacco}A / {fleet.n_lager}L")
    print(f"  Rotta: {fleet.delivery_route.value}")
    print(f"  Coating: {fleet.drone_spec.surface_coating}")
    print(f"\n  Parametri PK nel GBM:")
    print(f"    Penetrazione BEE:  {summary['bee_penetration']*100:.1f}%")
    print(f"    Cmax nel GBM:      {summary['cmax_gbm']:.4f} a.u.")
    print(f"    Tmax:              {summary['tmax_h']:.1f} h")
    print(f"    T½ stimato:        {summary['t_half_h']} h")
    print(f"    AUC GBM:           {summary['auc_gbm']:.4f}")
    print(f"    Picco effetto:     {summary['peak_effect']*100:.1f}%")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Comparazione farmaci
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 4: COMPARAZIONE NANODRONI vs FARMACI CLASSICI")
    comp = DrugComparison()
    comp_df = comp.compare()

    print("Ranking efficacy score (nanodroni vs farmaci convenzionali):")
    for _, row in comp_df.iterrows():
        tag = "▶ NANO" if "Nano" in row["agent"] else "  FARM"
        print(f"  {tag}  {row['agent']:<22} "
              f"score={row['efficacy_score']:.3f}  "
              f"BEE={row['bee_penetration']*100:.0f}%  "
              f"sel={row['selectivity']:.2f}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Genera visualizzazioni
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 5: GENERAZIONE VISUALIZZAZIONI")
    print("Generazione plot in corso...")

    p1 = plot_mutation_landscape(mut_df)
    print(f"  ✓ {p1.name}")

    p2 = plot_target_scores(score_df)
    print(f"  ✓ {p2.name}")

    p3 = plot_pk_simulation(pk_df, summary, fleet.delivery_route.value)
    print(f"  ✓ {p3.name}")

    p4 = plot_drug_comparison(comp_df)
    print(f"  ✓ {p4.name}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 6: Report testuale
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 6: REPORT FINALE")
    report = generate_report(mut_df, score_df, sl_df, summary, comp_df, top3_genes)
    report_path = Path(__file__).parent / "output" / "report_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  ✓ report_summary.txt")

    separator("PIPELINE COMPLETATA")
    print(f"\nOutput generati in: {Path(__file__).parent / 'output'}")
    print("\nConsigli terapeutici computazionali:")
    for _, row in top_targets.head(3).iterrows():
        print(f"  → {row['gene']}: {row['recommendation']}")

    print("\n⚠  DISCLAIMER: Simulazione computazionale a scopo di ricerca.")
    print("   Non costituisce raccomandazione clinica.")
    print("   Validazione richiesta: in vitro → in vivo → trial clinici.\n")


def generate_report(mut_df, score_df, sl_df, pk_summary, comp_df, top_genes):
    lines = []
    lines.append("═" * 68)
    lines.append("GBM COMPUTATIONAL PIPELINE — REPORT SOMMARIO")
    lines.append("Dati: TCGA-GBM | Brennan et al. Cell 2013")
    lines.append("═" * 68)

    lines.append("\n[1] PROFILO MUTAZIONALE GBM")
    lines.append(f"    Geni analizzati: {len(mut_df)}")
    lines.append(f"    Gene più mutato: {mut_df.iloc[0]['gene']} "
                 f"({mut_df.iloc[0]['frequency']}%)")
    lines.append(f"    Pathway dominante: RTK/RAS/PI3K (88.5% dei pazienti)")

    lines.append("\n[2] TARGET PRIORITARI")
    for _, row in score_df[score_df["druggable"]].head(5).iterrows():
        lines.append(f"    [{row['priority_rank']}] {row['gene']:<10} "
                     f"score={row['composite_score']:.3f}")

    lines.append("\n[3] LETHALITY SINTETICA (evidence HIGH)")
    for _, row in sl_df[sl_df["evidence"] == "HIGH"].iterrows():
        lines.append(f"    {row['mutated_gene']} mutato → target {row['target_gene']} "
                     f"({row['reference']})")

    lines.append("\n[4] PK NANODRONE (FUS-guided, 13 droni totali)")
    lines.append(f"    Penetrazione BEE: {pk_summary['bee_penetration']*100:.1f}%")
    lines.append(f"    Picco effetto:    {pk_summary['peak_effect']*100:.1f}%")
    lines.append(f"    T½ nel GBM:       {pk_summary['t_half_h']} h")

    lines.append("\n[5] AGENTE COMPUTAZIONALMENTE OTTIMALE")
    best = comp_df.iloc[0]
    lines.append(f"    {best['agent']} (score={best['efficacy_score']:.3f}, "
                 f"BEE={best['bee_penetration']*100:.0f}%)")

    lines.append("\n[6] PROSSIMI STEP CONSIGLIATI")
    lines.append("    1. Validazione in vitro su linee GBM (U87, U251, GSC)")
    lines.append("    2. Organoidi cerebrali 3D con profilo EGFR/PTEN del paziente")
    lines.append("    3. Test FUS su modello murin GBM per conferma apertura BEE")
    lines.append("    4. Sequenziamento singola cellula (scRNA-seq) per eterogeneità")
    lines.append("    5. Integrazione con AlphaFold per struttura proteine target")

    lines.append("\n⚠  Simulazione computazionale. Non per uso clinico.")
    lines.append("═" * 68)

    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline()
