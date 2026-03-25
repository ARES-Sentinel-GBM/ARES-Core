"""
modules/target_analyzer.py
===========================
Analisi target molecolari per GBM.
Scoring composito: frequenza mutazionale + espressione + druggability + BEE.
"""

import pandas as pd
import numpy as np

# ── BEE Penetration map per metodo di delivery ─────────────────────────────
BEE_PENETRATION = {
    "IV_free":        0.02,   # farmaco libero endovena (BEE intatta)
    "Liposome":       0.12,   # liposomi PEGylati
    "Nanoparticle":   0.25,   # NP polimeriche PLGA
    "Transferrin_NP": 0.45,   # NP + coating transferrina (RMT)
    "Exosome":        0.35,   # esosomi ingegnerizzati
    "FUS_NP":         0.68,   # NP + FUS (focused ultrasound apertura BEE)
    "FUS_Transferrin":0.72,   # NP + Transferrin + FUS (configurazione ottimale)
    "Intratumoral":   0.95,   # iniezione diretta (invasiva)
    "CED":            0.90,   # convection-enhanced delivery
}

# ── Mappa druggability per gene ────────────────────────────────────────────
_DRUGGABLE = {
    "EGFR": True,  "PDGFRA": True, "MET": True,   "VEGFA": True,
    "CDK4": True,  "MDM2": True,   "PIK3CA": True, "PIK3R1": True,
    "NF1":  False, "PTEN": False,  "RB1": False,  "TP53": False,
    "TERT": False, "CDKN2A": False,"ATRX": False,  "CIC": False,
    "FUBP1": False,"MDM4": True,   "MYCN": False,  "IDH1": True,
}

_RECOMMENDATIONS = {
    "EGFR":   "Osimertinib (3G TKI) + Nano-EGFR-Ab FUS-guided",
    "CDK4":   "Palbociclib + Ribociclib; nano-delivery per BEE",
    "MDM2":   "Idasanutlin (RG7388) + p53 reactivation combo",
    "PIK3CA": "Alpelisib; combo con MEK inh per feedback loop",
    "PDGFRA": "Avapritinib; nano-LNP per concentrazione tumorale",
    "MET":    "Capmatinib; FUS-NP per amplificazione MET",
    "MDM4":   "ALRN-6924 stapled peptide; dual MDM2/4 blockade",
    "IDH1":   "Ivosidenib; raramente mutato in IDH-wt GBM",
    "PIK3R1": "Alpelisib + mTOR inh; PI3K pathway dual block",
    "VEGFA":  "Bevacizumab + anti-ang2; combinazione anti-angiogenica",
}


class TargetAnalyzer:
    """
    Analisi e scoring dei target molecolari in GBM.

    Args:
        mut_df:  DataFrame mutazioni (gene, frequency, pathway, alteration_type)
        expr_df: DataFrame espressione (gene, zscore, log2fc, pval_adj)
    """

    def __init__(self, mut_df: pd.DataFrame, expr_df: pd.DataFrame):
        self.mut_df  = mut_df.copy()
        self.expr_df = expr_df.copy()
        self._expr_map = dict(zip(expr_df["gene"], expr_df["zscore"]))

    # ------------------------------------------------------------------ #
    def score_targets(self) -> pd.DataFrame:
        """
        Calcola composite score per ogni target.
        Score = 0.4*freq_norm + 0.35*expr_norm + 0.25*druggability
        """
        df = self.mut_df.copy()
        df["druggable"] = df["gene"].map(lambda g: _DRUGGABLE.get(g, False))

        # Normalizza frequenza [0,1]
        fmax = df["frequency"].max()
        df["freq_norm"] = df["frequency"] / fmax

        # Espressione z-score normalizzato [0,1]
        df["zscore"] = df["gene"].map(self._expr_map).fillna(0.0)
        z_abs = df["zscore"].abs()
        df["expr_norm"] = (z_abs - z_abs.min()) / (z_abs.max() - z_abs.min() + 1e-9)

        # Druggability score
        df["drug_score"] = df["druggable"].astype(float)

        # Composite score
        df["composite_score"] = (
            0.40 * df["freq_norm"] +
            0.35 * df["expr_norm"] +
            0.25 * df["drug_score"]
        ).round(4)

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["priority_rank"]  = df.index + 1
        df["mutation_freq"]  = df["frequency"]
        df["recommendation"] = df["gene"].map(
            lambda g: _RECOMMENDATIONS.get(g, f"Target {g}: ulteriore validazione richiesta")
        )

        cols = ["priority_rank", "gene", "composite_score", "mutation_freq",
                "pathway", "druggable", "zscore", "recommendation"]
        return df[cols]

    # ------------------------------------------------------------------ #
    def rank_pathways(self) -> pd.DataFrame:
        """
        Ranking dei pathway per mean composite score e numero geni druggabili.
        """
        scored = self.score_targets()
        grp = scored.groupby("pathway").agg(
            mean_score=("composite_score", "mean"),
            druggable_genes=("druggable", "sum"),
            total_genes=("gene", "count"),
            max_freq=("mutation_freq", "max"),
        ).reset_index()
        return grp.sort_values("mean_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def get_synthetic_lethality_map(self) -> pd.DataFrame:
        """
        Mappa di letalità sintetica basata su letteratura GBM.
        Fonte: Lord & Ashworth Nature 2017; Sajesh et al. 2020.
        """
        sl_pairs = [
            # mutated_gene  target_gene  evidence  ref
            ("PTEN",   "PARP1",    "HIGH",   "Mendes-Pereira NAT 2009"),
            ("PTEN",   "CHK1",     "HIGH",   "McCabe CCR 2015"),
            ("CDKN2A", "CDK4",     "HIGH",   "Wiedemeyer CCR 2010"),
            ("CDKN2A", "CDK6",     "HIGH",   "Wiedemeyer CCR 2010"),
            ("NF1",    "MEK1",     "HIGH",   "Lito Nat Med 2013"),
            ("TP53",   "MDM2",     "HIGH",   "Vassilev Science 2004"),
            ("RB1",    "EZH2",     "MEDIUM", "Knutson Cancer Cell 2013"),
            ("EGFR",   "MET",      "MEDIUM", "Stommel Science 2007"),
            ("IDH1",   "NAD+",     "MEDIUM", "Tateishi Nat Genet 2015"),
            ("ATRX",   "PARP",     "MEDIUM", "Flynn EMBO 2015"),
        ]
        df = pd.DataFrame(sl_pairs, columns=[
            "mutated_gene", "target_gene", "evidence", "reference"
        ])
        # Aggiungi frequenza mutazionale
        freq_map = dict(zip(self.mut_df["gene"], self.mut_df["frequency"]))
        df["mut_frequency"] = df["mutated_gene"].map(freq_map).fillna(0.0)
        return df.sort_values(["evidence", "mut_frequency"], ascending=[True, False])

    # ------------------------------------------------------------------ #
    def bee_strategy_matrix(self, genes: list) -> pd.DataFrame:
        """
        Matrice strategie di delivery per i geni selezionati.
        Ordina per penetrazione BEE decrescente.
        """
        rows = []
        for method, bee in BEE_PENETRATION.items():
            rows.append({
                "target_genes":     ", ".join(genes),
                "delivery_method":  method,
                "bee_penetration":  bee,
                "invasiveness":     "invasive" if bee > 0.85 else "non-invasive",
                "feasibility":      "clinical" if bee < 0.80 else "experimental",
                "recommended":      bee >= 0.60,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("bee_penetration", ascending=False).reset_index(drop=True)
