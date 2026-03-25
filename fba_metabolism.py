"""
modules/fba_metabolism.py  — GBM Flux Balance Analysis
========================================================
Modello metabolico GBM minimo ma corretto (17 rx, 10 met interni).
Solver: scipy.optimize.linprog (HiGHS).

S · v = 0  (steady-state)
lb ≤ v ≤ ub
max  c^T · v  (obiettivo: biomassa tumorale)

Ref:
  Orth et al. Nat Biotechnol 2010
  Vander Heiden et al. Science 2009 (Warburg)
  Mashimo et al. Science 2014 (GBM metabolism)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import linprog

# ── Metaboliti interni ────────────────────────────────────────────
# Cofattori semplificati: un pool ATP e un pool NADH
METS = ["g6p", "pyr", "lac", "acetcoa", "akg",
        "r5p", "lipid", "biomass", "ATP", "NADH"]
M = len(METS); mi = {m: i for i, m in enumerate(METS)}

# ── Reazioni ──────────────────────────────────────────────────────
RXNS = [
    # Exchange (bounds controllano uptake/secrezione)
    "EX_glc",    # glucosio uptake  → g6p + ATP (GLUT1 + HK lumped)
    "EX_gln",    # glutammina uptake → akg (GLS + GLUD lumped)
    "EX_lac",    # lattato export   (Warburg)
    "EX_bio",    # biomassa export  (OBIETTIVO)
    # Glicolisi (Warburg: produce lattato invece di entrare in TCA)
    "GLYC",      # g6p → 2 pyr + 2 ATP
    "LDH",       # pyr → lac + NADH_consumed (NADH → NAD, sink NADH)
    # Piruvato → TCA
    "PDH",       # pyr → acetcoa + NADH
    "TCA",       # acetcoa + akg_sink → 3 NADH + 2 ATP
    # PPP e biosintesi
    "PPP",       # g6p → r5p + NADH
    "FASN",      # acetcoa → lipid  (consuma NADH)
    # Fosforilazione ossidativa
    "OXPHOS",    # NADH → 2.5 ATP
    # Reazione biomassa
    "BIOMASS",   # g6p + r5p + lipid + ATP → biomass
    # Sink obbligatori per bilanciamento
    "ATP_sink",  # ATP → (domanda energetica cellulare)
    "NADH_sink", # NADH → (ossidazione residua, overflow)
    "PYR_sink",  # pyr → (overflow piruvato)
    "AKG_sink",  # akg → (overflow akg)
    "R5P_sink",  # r5p → (sink nucleotidi)
]
N = len(RXNS); ri = {r: j for j, r in enumerate(RXNS)}

# Gene → Reazione per knockout
GENE_RXN: Dict[str, List[str]] = {
    "GLUT1": ["EX_glc"],  "HK2":  ["EX_glc"],
    "PFKM":  ["GLYC"],    "PFKL": ["GLYC"],
    "LDHA":  ["LDH"],     "MCT4": ["EX_lac"],
    "PDHA1": ["PDH"],     "IDH1": ["TCA"],
    "GLS":   ["EX_gln"],  "G6PD": ["PPP"],
    "FASN":  ["FASN"],    "SDHA": ["TCA"],
}
GENE_DRUGS: Dict[str, Dict] = {
    "HK2":  {"drug":"2-DG",       "stage":"pre-clinical","ref":"Maher 2004"},
    "PFKM": {"drug":"3PO",        "stage":"phase I",     "ref":"Clem 2008"},
    "LDHA": {"drug":"FX11",       "stage":"pre-clinical","ref":"Le 2010"},
    "MCT4": {"drug":"AZD3965",    "stage":"phase I",     "ref":"Polanski 2014"},
    "GLS":  {"drug":"CB-839",     "stage":"phase II",    "ref":"Gross 2014"},
    "FASN": {"drug":"TVB-2640",   "stage":"phase II",    "ref":"Kuhajda 2006"},
    "IDH1": {"drug":"Ivosidenib", "stage":"approved",    "ref":"Rohle 2013"},
    "G6PD": {"drug":"6-AN",       "stage":"pre-clinical","ref":"Budihardjo 1998"},
}

def _S() -> np.ndarray:
    S = np.zeros((M, N))

    # EX_glc: → g6p + ATP  (uptake glucosio + HK)
    S[mi["g6p"], ri["EX_glc"]]  = +1
    S[mi["ATP"], ri["EX_glc"]]  = +1   # netto: HK usa ATP ma GLYC ne produce 2

    # EX_gln: → akg  (GLS + GLUD)
    S[mi["akg"], ri["EX_gln"]]  = +1

    # EX_lac: lac →
    S[mi["lac"], ri["EX_lac"]]  = -1

    # EX_bio: biomass →
    S[mi["biomass"], ri["EX_bio"]] = -1

    # GLYC: g6p → 2 pyr + 2 ATP  (glicolisi)
    S[mi["g6p"],  ri["GLYC"]]   = -1
    S[mi["pyr"],  ri["GLYC"]]   = +2
    S[mi["ATP"],  ri["GLYC"]]   = +2

    # LDH: pyr → lac  (consuma NADH netto: sink NADH tramite NAD rigenerazione)
    S[mi["pyr"],  ri["LDH"]]    = -1
    S[mi["lac"],  ri["LDH"]]    = +1
    S[mi["NADH"], ri["LDH"]]    = -1   # LDH consuma NADH

    # PDH: pyr → acetcoa + NADH
    S[mi["pyr"],    ri["PDH"]]  = -1
    S[mi["acetcoa"],ri["PDH"]]  = +1
    S[mi["NADH"],   ri["PDH"]]  = +1

    # TCA: acetcoa + akg → 3 NADH + 2 ATP  (TCA + GLUD lumped)
    S[mi["acetcoa"],ri["TCA"]]  = -1
    S[mi["akg"],    ri["TCA"]]  = -1
    S[mi["NADH"],   ri["TCA"]]  = +3
    S[mi["ATP"],    ri["TCA"]]  = +2

    # PPP: g6p → r5p + NADH
    S[mi["g6p"],  ri["PPP"]]    = -1
    S[mi["r5p"],  ri["PPP"]]    = +1
    S[mi["NADH"], ri["PPP"]]    = +1

    # FASN: acetcoa → lipid  (consuma NADH)
    S[mi["acetcoa"],ri["FASN"]] = -2
    S[mi["lipid"],  ri["FASN"]] = +1
    S[mi["NADH"],   ri["FASN"]] = -1

    # OXPHOS: NADH → 2.5 ATP
    S[mi["NADH"], ri["OXPHOS"]] = -1
    S[mi["ATP"],  ri["OXPHOS"]] = +2

    # BIOMASS: g6p + r5p + lipid + 2 ATP → biomass
    S[mi["g6p"],    ri["BIOMASS"]] = -0.3
    S[mi["r5p"],    ri["BIOMASS"]] = -0.2
    S[mi["lipid"],  ri["BIOMASS"]] = -0.2
    S[mi["ATP"],    ri["BIOMASS"]] = -2.0
    S[mi["biomass"],ri["BIOMASS"]] = +1.0

    # Sinks
    S[mi["ATP"],  ri["ATP_sink"]]  = -1
    S[mi["NADH"], ri["NADH_sink"]] = -1
    S[mi["pyr"],  ri["PYR_sink"]]  = -1
    S[mi["akg"],  ri["AKG_sink"]]  = -1
    S[mi["r5p"],  ri["R5P_sink"]]  = -1

    return S

def _bounds(wf: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.zeros(N)
    ub = np.full(N, 1000.0)
    ub[ri["EX_glc"]]   = 20.0 * wf       # Warburg: alta glicolisi
    ub[ri["EX_gln"]]   = 10.0
    ub[ri["EX_lac"]]   = 30.0 * wf       # alta secrezione lattato
    ub[ri["LDH"]]      = 25.0 * wf       # LDH-A upregolata
    ub[ri["PDH"]]      = 5.0 / max(wf,0.5)  # PDH soppressa in Warburg
    ub[ri["OXPHOS"]]   = 15.0 / max(wf,0.5) # OXPHOS ridotta
    ub[ri["FASN"]]     = 8.0 * wf
    return lb, ub


class GBMMetabolicModel:
    """FBA per metabolismo GBM. max biomassa s.t. S·v=0, lb≤v≤ub."""

    def __init__(self, warburg_factor: float = 1.0):
        self.wf = warburg_factor
        self.S  = _S()
        self.lb, self.ub = _bounds(warburg_factor)

    # ── FBA ──────────────────────────────────────────────────────────
    def solve_fba(self) -> dict:
        c = np.zeros(N); c[ri["EX_bio"]] = -1.0
        res = linprog(c, A_eq=self.S, b_eq=np.zeros(M),
                      bounds=list(zip(self.lb, self.ub)),
                      method="highs", options={"disp": False})
        if res.status == 0:
            v = res.x
            active = sorted([(RXNS[j], round(float(v[j]),4))
                              for j in range(N) if abs(v[j]) > 1e-6],
                             key=lambda x: -abs(x[1]))
            return {"status":"optimal",
                    "biomass": round(float(v[ri["EX_bio"]]),6),
                    "v": v, "active": active,
                    "sv_max": float(np.max(np.abs(self.S @ v)))}
        return {"status": res.message, "biomass": 0.0,
                "v": np.zeros(N), "active": [], "sv_max": 0.0}

    # ── Gene Essentiality ─────────────────────────────────────────────
    def gene_essentiality(self, threshold: float = 0.05) -> pd.DataFrame:
        wt = self.solve_fba()
        if wt["status"] != "optimal" or wt["biomass"] < 1e-8:
            return pd.DataFrame()
        wt_bio = wt["biomass"]
        rows = []
        for gene, rxns in GENE_RXN.items():
            orig = [(r, float(self.lb[ri[r]]), float(self.ub[ri[r]])) for r in rxns]
            for r in rxns: self.lb[ri[r]] = 0; self.ub[ri[r]] = 0
            ko = self.solve_fba()
            for r, l, u in orig: self.lb[ri[r]] = l; self.ub[ri[r]] = u
            ratio = ko["biomass"] / (wt_bio + 1e-12)
            drug  = GENE_DRUGS.get(gene, {})
            rows.append({
                "gene": gene, "wt_biomass": round(wt_bio,4),
                "ko_biomass": round(ko["biomass"],4),
                "ratio": round(ratio,4),
                "essential": ratio < threshold,
                "drug":  drug.get("drug","—"),
                "stage": drug.get("stage","—"),
                "ref":   drug.get("ref","—"),
            })
        df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
        df["priority"] = df.apply(
            lambda r: "HIGH" if r["essential"] and r["drug"]!="—"
                      else "MED"  if r["essential"] else "LOW", axis=1)
        return df

    # ── Warburg scan ─────────────────────────────────────────────────
    def warburg_scan(self, factors=None) -> pd.DataFrame:
        if factors is None:
            factors = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        rows = []
        for wf in factors:
            m = GBMMetabolicModel(wf); res = m.solve_fba(); v = res["v"]
            rows.append({
                "warburg_factor": wf,
                "biomass":        res["biomass"],
                "glycolysis":     round(float(v[ri["GLYC"]]),3),
                "ldh_warburg":    round(float(v[ri["LDH"]]),3),
                "tca":            round(float(v[ri["TCA"]]),3),
                "oxphos":         round(float(v[ri["OXPHOS"]]),3),
                "lactate_export": round(float(v[ri["EX_lac"]]),3),
            })
        return pd.DataFrame(rows)

    # ── Flux summary table ────────────────────────────────────────────
    def flux_summary(self) -> pd.DataFrame:
        v = self.solve_fba()["v"]
        return pd.DataFrame([{"reaction": RXNS[j], "flux": round(float(v[j]),4)}
                              for j in range(N)]).sort_values("flux", ascending=False)
