"""
run_alphafold.py
----------------
Modulo 2 della pipeline GBM: AlphaFold + Docking.

Esegui come:
    python run_alphafold.py

Oppure importa come modulo:
    from run_alphafold import run_alphafold_pipeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from data.alphafold_data import (
    BINDING_POCKETS, ALPHAFOLD_PLDDT, DRUG_PHYSICOCHEMISTRY,
    AlphaFoldConnector
)
from modules.docking_engine import (
    VirtualScreeningEngine, analyze_pocket_geometry,
    BindingScorer, BEEPermeabilityPredictor, ResistanceAnalyzer
)
from modules.alphafold_viz import (
    plot_plddt_profiles, plot_docking_results, plot_best_candidates
)
from pathlib import Path


def separator(title=""):
    w = 68
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * w)


def run_alphafold_pipeline():
    print("\n" + "═"*68)
    print("  GBM PIPELINE — MODULO 2: ALPHAFOLD + DOCKING MOLECOLARE")
    print("  Strutture reali da PDB/AlphaFold | Scoring empirico calibrato")
    print("═"*68)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Fetch info strutturali AlphaFold
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 1: STRUTTURE ALPHAFOLD")

    connector = AlphaFoldConnector()
    target_genes = ["EGFR", "PIK3CA", "CDK4", "MDM2", "MET"]

    print("Strutture AlphaFold per target GBM:")
    for gene in target_genes:
        info = connector.fetch_structure_info(gene)
        plddt = ALPHAFOLD_PLDDT.get(gene, {}).get("overall_mean", 0)
        print(f"  {gene:<10} UniProt: {info['uniprot_id']}  "
              f"pLDDT_mean={plddt:.1f}  "
              f"[{info['source']}]")
        print(f"           ↳ {info['pdb_url']}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Analisi geometrica pocket
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 2: ANALISI GEOMETRICA BINDING POCKETS")

    pocket_analyses = []
    for gene in BINDING_POCKETS:
        for pocket_name in BINDING_POCKETS[gene]:
            geo = analyze_pocket_geometry(gene, pocket_name)
            pocket_analyses.append(geo)

    print(f"\n{'Gene':<10} {'Pocket':<22} {'Vol(Å³)':<10} {'Drug.':<8} {'nRes':<6} {'Hydrophob':<10}")
    print("─" * 70)
    for p in sorted(pocket_analyses, key=lambda x: x.get("druggability_score", 0), reverse=True):
        print(f"  {p['gene']:<8} {p['pocket']:<22} "
              f"{p['volume_A3']:<10.0f} {p['druggability_score']:<8.2f} "
              f"{p['n_residues']:<6} {p['hydrophobic_ratio']:.0%}")

    # Hotspot e gatekeeper
    print("\nHotspot e gatekeeper critici:")
    for p in pocket_analyses:
        if p.get("hotspot", "N/A") != "N/A":
            print(f"  {p['gene']}/{p['pocket']}: "
                  f"hotspot={p['hotspot']}  "
                  f"gatekeeper={p.get('gatekeeper','N/A')}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: BEE Permeability dei farmaci
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 3: CNS MPO — PERMEABILITÀ BEE FARMACI")

    bee_pred = BEEPermeabilityPredictor()
    drugs = list(DRUG_PHYSICOCHEMISTRY.keys())

    print(f"\n{'Farmaco':<16} {'MW':<7} {'logP':<7} {'TPSA':<7} {'HBD':<5} {'CNS_MPO':<9} {'BEE%':<8} {'Profilo'}")
    print("─" * 80)
    for drug in drugs:
        pc = DRUG_PHYSICOCHEMISTRY[drug]
        mpo, bee = bee_pred.score(drug)
        profilo = "✓ CNS-ok" if mpo >= 4 else "⚠ Marginale" if mpo >= 3 else "✗ Scarso"
        print(f"  {drug:<14} {pc['MW']:<7.0f} {pc['logP']:<7.1f} "
              f"{pc['TPSA']:<7.0f} {pc['HBD']:<5} {mpo:<9.1f} "
              f"{bee*100:<8.1f} {profilo}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Virtual Screening (tutti farmaci vs tutti target)
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 4: VIRTUAL SCREENING DOCKING")

    engine = VirtualScreeningEngine()
    screen_genes = list(BINDING_POCKETS.keys())
    screen_drugs = drugs

    print(f"\nScreening {len(screen_drugs)} farmaci × {len(screen_genes)} target × pockets...")
    dock_df = engine.screen(screen_genes, screen_drugs)
    print(f"✓ Combinazioni valutate: {len(dock_df)}")

    print(f"\nTop 10 candidati per overall score:")
    print(f"{'Rank':<5} {'Drug':<16} {'Gene':<8} {'Pocket':<22} "
          f"{'ΔG':<8} {'Kd(nM)':<10} {'BEE%':<7} {'Score':<7} {'Resist'}")
    print("─" * 92)
    for _, row in dock_df.head(10).iterrows():
        print(f"  [{row['rank']:>2}] {row['drug']:<14} {row['gene']:<8} "
              f"{row['pocket']:<22} "
              f"{row['binding_score']:<8.1f} {row['Kd_predicted_nM']:<10.2f} "
              f"{row['bee_penetration']*100:<7.1f} "
              f"{row['overall_score']:<7.3f} {row['resistance_risk']}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Analisi resistenze
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 5: ANALISI MUTAZIONI DI RESISTENZA")

    resist = ResistanceAnalyzer()
    key_pairs = [
        ("erlotinib",   "EGFR"),
        ("palbociclib", "CDK4"),
        ("milademetan", "MDM2"),
        ("BKM120",      "PIK3CA"),
    ]
    for drug, gene in key_pairs:
        r = resist.evaluate(drug, gene)
        print(f"\n  {drug} → {gene}:")
        print(f"    Rischio: {r['resistance_risk']}  |  ΔΔG: {r['total_dg_penalty_kcal']:+.1f} kcal/mol")
        print(f"    {r['recommendation']}")
        for mut, info in r["mutations_analyzed"].items():
            print(f"    · {mut}: {info['mechanism']}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 6: Visualizzazioni
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 6: GENERAZIONE VISUALIZZAZIONI")

    p5 = plot_plddt_profiles(ALPHAFOLD_PLDDT)
    print(f"  ✓ {p5.name}")

    p6 = plot_docking_results(dock_df)
    print(f"  ✓ {p6.name}")

    p7 = plot_best_candidates(dock_df)
    print(f"  ✓ {p7.name}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 7: Sintesi finale
    # ──────────────────────────────────────────────────────────────────
    separator("SINTESI — CANDIDATI TERAPEUTICI")

    best = dock_df.head(3)
    print("\nTop 3 candidati computazionali per GBM:")
    for _, row in best.iterrows():
        print(f"\n  ★ {row['drug']} → {row['gene']} ({row['pocket']})")
        print(f"    ΔG={row['binding_score']:.1f} kcal/mol | Kd≈{row['Kd_predicted_nM']:.1f} nM")
        print(f"    BEE penetrazione: {row['bee_penetration']*100:.0f}%")
        print(f"    Fiducia struttura (pLDDT): {row['plddt_pocket']:.0f}")
        print(f"    {row['recommendation']}")

    print("\nProssimi step consigliati:")
    steps = [
        "Download strutture PDB da AlphaFold EBI (URLs già generati)",
        "Docking ad alta risoluzione con AutoDock Vina o Glide (Schrödinger)",
        "MD simulation (GROMACS/AMBER) per validare stabilità complesso",
        "Test in vitro: IC50 su linee GBM (U87, U251, GSC) con candidati top",
        "Profilo ADMET completo con SwissADME/pkCSM per candidati validati",
        "Organoidi cerebrali 3D: test penetrazione BEE su modello ex vivo",
    ]
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    separator()
    print("⚠  Simulazione computazionale a scopo di ricerca.")
    print("   Validazione sperimentale obbligatoria prima di qualsiasi uso clinico.\n")

    return dock_df


if __name__ == "__main__":
    dock_df = run_alphafold_pipeline()
