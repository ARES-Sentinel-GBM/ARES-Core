"""
modules/vitro_predictor.py
===========================
Predittore computazionale di risposta in vitro per linee cellulari GBM.

Linee supportate:
  U87    — GBM classico, EGFR+, PTEN-null, p53-wt
  U251   — GBM mesenchimale, TP53 mut, EGFR amp, MGMT-
  LN229  — GBM proneural, PTEN mut, TP53 mut
  T98G   — GBM recidivante, multifarmacoresistente
  GSC    — Glioblastoma Stem Cells (sferoidi neurosphere)
  U87-EGFRvIII — U87 con variante EGFRvIII (spesso TMZ-resistente)

Il modello predice:
  - IC50 nanodrone vs farmaco classico
  - Curva dose-risposta (Hill equation)
  - Indice di selettività (GBM vs astrociti normali)
  - Probabilità di drug resistance

Fonte parametri: Reardon et al. Neuro-Oncol 2011; Affronti et al. 2018;
                 McLendon et al. Nature 2008 (TCGA pilot).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


# ── Profili linee cellulari ────────────────────────────────────────────────────
CELL_LINE_PROFILES = {
    "U87": {
        "subtype":        "Classical",
        "egfr_status":    "overexpressed",
        "pten_status":    "null",
        "tp53_status":    "wildtype",
        "mgmt_status":    "unmethylated",
        "idh_status":     "wildtype",
        "doubling_time_h": 24,
        "base_sensitivity": {
            "TMZ":          0.45,
            "Bevacizumab":  0.30,
            "Erlotinib":    0.55,
            "Carmustine":   0.40,
            "Nano-EGFR":    0.78,
            "Nano-TMZ-FUS": 0.82,
        },
        "normal_selectivity_factor": 3.2,
    },
    "U251": {
        "subtype":        "Mesenchymal",
        "egfr_status":    "amplified",
        "pten_status":    "mutant",
        "tp53_status":    "mutant",
        "mgmt_status":    "unmethylated",
        "idh_status":     "wildtype",
        "doubling_time_h": 36,
        "base_sensitivity": {
            "TMZ":          0.35,
            "Bevacizumab":  0.25,
            "Erlotinib":    0.40,
            "Carmustine":   0.38,
            "Nano-EGFR":    0.72,
            "Nano-TMZ-FUS": 0.76,
        },
        "normal_selectivity_factor": 4.1,
    },
    "LN229": {
        "subtype":        "Proneural",
        "egfr_status":    "wildtype",
        "pten_status":    "mutant",
        "tp53_status":    "mutant",
        "mgmt_status":    "methylated",
        "idh_status":     "wildtype",
        "doubling_time_h": 28,
        "base_sensitivity": {
            "TMZ":          0.65,   # MGMT-methylated → più sensibile a TMZ
            "Bevacizumab":  0.35,
            "Erlotinib":    0.35,
            "Carmustine":   0.50,
            "Nano-EGFR":    0.60,
            "Nano-TMZ-FUS": 0.80,
        },
        "normal_selectivity_factor": 2.8,
    },
    "T98G": {
        "subtype":        "Mesenchymal",
        "egfr_status":    "amplified",
        "pten_status":    "mutant",
        "tp53_status":    "mutant",
        "mgmt_status":    "unmethylated",
        "idh_status":     "wildtype",
        "doubling_time_h": 32,
        "base_sensitivity": {
            "TMZ":          0.20,   # resistente
            "Bevacizumab":  0.18,
            "Erlotinib":    0.25,
            "Carmustine":   0.30,
            "Nano-EGFR":    0.65,
            "Nano-TMZ-FUS": 0.70,
        },
        "normal_selectivity_factor": 5.0,
    },
    "GSC": {
        "subtype":        "Stem-like",
        "egfr_status":    "variable",
        "pten_status":    "variable",
        "tp53_status":    "variable",
        "mgmt_status":    "variable",
        "idh_status":     "wildtype",
        "doubling_time_h": 72,     # sferoidi crescono più lentamente
        "base_sensitivity": {
            "TMZ":          0.25,   # stem cells spesso resistenti
            "Bevacizumab":  0.20,
            "Erlotinib":    0.30,
            "Carmustine":   0.28,
            "Nano-EGFR":    0.60,
            "Nano-TMZ-FUS": 0.68,
        },
        "normal_selectivity_factor": 6.2,
    },
    "U87-EGFRvIII": {
        "subtype":        "Classical/EGFRvIII",
        "egfr_status":    "EGFRvIII_constitutive",
        "pten_status":    "null",
        "tp53_status":    "wildtype",
        "mgmt_status":    "unmethylated",
        "idh_status":     "wildtype",
        "doubling_time_h": 22,
        "base_sensitivity": {
            "TMZ":          0.40,
            "Bevacizumab":  0.28,
            "Erlotinib":    0.30,   # EGFRvIII non risponde bene a TKI convenzionali
            "Carmustine":   0.38,
            "Nano-EGFR":    0.85,   # anticorpo anti-EGFR ottimale per EGFRvIII
            "Nano-TMZ-FUS": 0.84,
        },
        "normal_selectivity_factor": 3.5,
    },
}


@dataclass
class DoseResponseResult:
    agent:       str
    cell_line:   str
    ic50_uM:     float
    hill_coef:   float   # coefficiente Hill (cooperatività)
    emax:        float   # effetto massimo [0-1]
    si:          float   # Selectivity Index (IC50_normal / IC50_tumor)
    resist_prob: float   # probabilità di drug resistance [0-1]


class VitroPredictor:
    """
    Predittore computazionale di risposta in vitro per linee GBM.

    Usa equazione di Hill per curve dose-risposta:
      E(C) = Emax * C^n / (IC50^n + C^n)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Curva dose-risposta ──────────────────────────────────────────────────
    def dose_response_curve(
        self,
        agent:      str,
        cell_line:  str,
        doses:      np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Genera curva dose-risposta per un agente su una linea cellulare.
        Returns: DataFrame (dose_uM, viability, effect)
        """
        if doses is None:
            doses = np.logspace(-3, 2, 50)   # 0.001 → 100 µM

        profile = CELL_LINE_PROFILES.get(cell_line)
        if not profile:
            raise ValueError(f"Cell line '{cell_line}' not found. "
                             f"Available: {list(CELL_LINE_PROFILES.keys())}")

        sens      = profile["base_sensitivity"].get(agent, 0.40)
        noise     = self.rng.normal(0, 0.02)

        # Parametri Hill
        emax      = min(0.98, sens + 0.15 + noise)
        ic50      = 10**(2 - sens * 3.5 + self.rng.normal(0, 0.1))   # µM
        n         = 1.2 + self.rng.uniform(-0.2, 0.8)                 # Hill coef

        effect    = emax * doses**n / (ic50**n + doses**n)
        viability = 1.0 - effect

        return pd.DataFrame({
            "dose_uM":   doses,
            "viability": np.clip(viability, 0, 1),
            "effect":    np.clip(effect, 0, 1),
            "agent":     agent,
            "cell_line": cell_line,
        })

    # ── IC50 e parametri per tutti gli agenti × linee ────────────────────────
    def compute_ic50_matrix(
        self,
        agents:     list = None,
        cell_lines: list = None,
    ) -> pd.DataFrame:
        """
        Calcola matrice IC50 per combinazioni agente × linea cellulare.
        """
        if agents     is None: agents     = list(CELL_LINE_PROFILES["U87"]["base_sensitivity"].keys())
        if cell_lines is None: cell_lines = list(CELL_LINE_PROFILES.keys())

        rows = []
        for cl in cell_lines:
            profile = CELL_LINE_PROFILES[cl]
            for ag in agents:
                sens  = profile["base_sensitivity"].get(ag, 0.35)
                noise = self.rng.normal(0, 0.03)
                emax  = min(0.98, sens + 0.15 + noise)
                ic50  = round(10**(2 - sens*3.5 + self.rng.normal(0, 0.15)), 3)
                n_hill= round(1.2 + self.rng.uniform(-0.2, 0.8), 2)

                # Selectivity Index
                si_factor = profile["normal_selectivity_factor"]
                ic50_normal = round(ic50 * si_factor, 3)
                si = round(ic50_normal / (ic50 + 1e-9), 2)

                # Resistenza: inversamente proporzionale a sensibilità
                resist_prob = round(max(0.0, (0.7 - sens) + self.rng.uniform(-0.1, 0.1)), 3)

                rows.append({
                    "cell_line":    cl,
                    "subtype":      profile["subtype"],
                    "agent":        ag,
                    "IC50_uM":      ic50,
                    "IC50_normal_uM": ic50_normal,
                    "emax":         round(emax, 3),
                    "hill_coef":    n_hill,
                    "SI":           si,
                    "resist_prob":  resist_prob,
                    "is_nano":      "Nano" in ag,
                    "mgmt":         profile["mgmt_status"],
                    "egfr":         profile["egfr_status"],
                })

        return pd.DataFrame(rows).sort_values(["cell_line","IC50_uM"]).reset_index(drop=True)

    # ── Raccomandazione per paziente ─────────────────────────────────────────
    def patient_recommendation(
        self,
        mutations:  list,       # lista gene mutati del paziente
        mgmt_status: str = "unmethylated",
    ) -> dict:
        """
        Raccomandazione farmacologica personalizzata basata su profilo mutazionale.
        """
        recs = []
        has_egfr  = "EGFR" in mutations
        has_pten  = "PTEN" in mutations
        has_cdkn2a= "CDKN2A" in mutations
        mgmt_meth = mgmt_status == "methylated"

        if has_egfr:
            recs.append({"agent": "Nano-EGFR", "rationale": "EGFR mutato/amplificato",
                          "priority": "HIGH", "evidence": "TCGA freq 57.4%"})
        if mgmt_meth:
            recs.append({"agent": "Nano-TMZ-FUS", "rationale": "MGMT methylato → TMZ sensibile",
                          "priority": "HIGH", "evidence": "Stupp NEJM 2005"})
        if has_cdkn2a:
            recs.append({"agent": "Palbociclib", "rationale": "CDKN2A loss → CDK4/6 dipendenza",
                          "priority": "MEDIUM", "evidence": "Wiedemeyer CCR 2010"})
        if has_pten:
            recs.append({"agent": "Olaparib", "rationale": "PTEN loss → PARP-SL",
                          "priority": "MEDIUM", "evidence": "Mendes-Pereira 2009"})
        if not recs:
            recs.append({"agent": "Nano-TMZ-FUS", "rationale": "Default per GBM IDH-wt",
                          "priority": "STANDARD", "evidence": "Stupp 2005"})

        return {
            "top_recommendation":  recs[0]["agent"],
            "all_recommendations": recs,
            "mgmt_benefit_tmz":   mgmt_meth,
            "n_targetable":       len(recs),
        }
