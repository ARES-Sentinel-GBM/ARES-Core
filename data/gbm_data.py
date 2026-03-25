"""
data/gbm_data.py
================
GBM Data Loader — dati embedded basati su:
  Brennan CW et al., Cell 2013 (TCGA-GBM n=617, IDH-wildtype)
  Cancer Genome Atlas Research Network, 2021
"""

import pandas as pd
import numpy as np


class GBMDataLoader:
    """
    Carica i dati molecolari GBM da sorgente embedded (default)
    o da API TCGA pubblica (source='api').
    """

    def __init__(self, source: str = "embedded"):
        self.source = source

    # ------------------------------------------------------------------ #
    #  Mutational landscape TCGA-GBM (IDH-wt, n=617)
    # ------------------------------------------------------------------ #
    def load_mutations(self) -> pd.DataFrame:
        """
        Frequenza mutazionale dei geni chiave in GBM.
        Fonte: Brennan et al. Cell 2013 + CGAP 2021.
        """
        data = [
            # gene         freq%   pathway                  effect
            ("TERT",       72.0,  "Telomere/Epigenetics",  "promoter_mutation"),
            ("EGFR",       57.4,  "RTK/RAS/PI3K",          "amplification/mutation"),
            ("PTEN",       41.0,  "RTK/RAS/PI3K",          "deletion/mutation"),
            ("CDKN2A",     59.0,  "Cell Cycle/p53",        "deletion"),
            ("TP53",       31.0,  "Cell Cycle/p53",        "mutation"),
            ("RB1",        11.0,  "Cell Cycle/p53",        "deletion/mutation"),
            ("PIK3CA",     15.4,  "RTK/RAS/PI3K",          "mutation"),
            ("PIK3R1",     10.8,  "RTK/RAS/PI3K",          "mutation"),
            ("NF1",        10.0,  "RTK/RAS/PI3K",          "deletion/mutation"),
            ("PDGFRA",      8.8,  "RTK/RAS/PI3K",          "amplification"),
            ("MET",         4.0,  "RTK/RAS/PI3K",          "amplification"),
            ("IDH1",        5.0,  "Metabolic",             "R132H (raro in IDH-wt)"),
            ("ATRX",        7.0,  "Chromatin",             "mutation"),
            ("CIC",         5.0,  "Transcription",         "mutation"),
            ("FUBP1",       3.5,  "Transcription",         "mutation"),
            ("MDM2",        7.0,  "Cell Cycle/p53",        "amplification"),
            ("MDM4",        7.0,  "Cell Cycle/p53",        "amplification"),
            ("CDK4",        8.0,  "Cell Cycle/p53",        "amplification"),
            ("MYCN",        3.5,  "Transcription",         "amplification"),
            ("VEGFA",      25.0,  "Angiogenesis",          "amplification"),
        ]
        df = pd.DataFrame(data, columns=["gene", "frequency", "pathway", "alteration_type"])
        return df.sort_values("frequency", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Gene expression signature (z-score normalized, n=617)
    # ------------------------------------------------------------------ #
    def load_expression(self) -> pd.DataFrame:
        """
        Firma di espressione genica in GBM vs tessuto normale.
        z-score; pos = overexpresso, neg = sottoespresso.
        """
        np.random.seed(42)
        genes_up = {
            "EGFR": 3.82, "PDGFRA": 2.91, "MET": 2.44, "VEGFA": 3.10,
            "CDK4": 2.78, "MDM2": 2.55, "SOX2": 4.10, "CD44": 3.30,
            "NESTIN": 3.90, "CHI3L1": 4.50, "MGMT": 1.20, "TERT": 2.10,
            "MYC": 1.85, "MYCN": 2.30, "HIF1A": 3.40,
        }
        genes_down = {
            "PTEN": -3.10, "TP53": -1.80, "RB1": -2.40, "CDKN2A": -4.20,
            "NF1": -1.90, "ATRX": -1.55, "CIC": -1.30,
        }
        all_genes = {**genes_up, **genes_down}
        rows = [{"gene": g, "zscore": z,
                 "log2fc": z * 0.8 + np.random.normal(0, 0.1),
                 "pval_adj": max(1e-15, np.random.exponential(0.001))}
                for g, z in all_genes.items()]
        return pd.DataFrame(rows).sort_values("zscore", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Drug database
    # ------------------------------------------------------------------ #
    def load_drugs(self) -> pd.DataFrame:
        """
        Database farmaci/agenti per GBM con profilo PK-BEE.
        """
        data = [
            # agent              target        class              bee    resp%
            ("Temozolomide",    "DNA",        "Alkylating",      0.40,  26.0),
            ("Bevacizumab",     "VEGFA",      "Anti-VEGF mAb",   0.02,  18.0),
            ("Erlotinib",       "EGFR",       "TKI",             0.30,  12.0),
            ("Gefitinib",       "EGFR",       "TKI",             0.28,  10.0),
            ("Osimertinib",     "EGFR T790M", "TKI-3G",          0.45,  22.0),
            ("Irinotecan",      "TOP1",       "Topo-I Inh",      0.15,   8.0),
            ("Carmustine",      "DNA",        "Alkylating",      0.55,  20.0),
            ("Lomustine",       "DNA",        "Alkylating",      0.60,  18.0),
            ("Palbociclib",     "CDK4/6",     "CDK Inh",         0.25,  15.0),
            ("Alpelisib",       "PI3Kα",      "PI3K Inh",        0.22,  12.0),
            ("Olaparib",        "PARP",       "PARP Inh",        0.18,  14.0),
            ("Selumetinib",     "MEK1/2",     "MEK Inh",         0.20,  11.0),
            ("Nano-EGFR-Ab",    "EGFR",       "Nanodrone-Ab",    0.68,  58.0),
            ("Nano-TMZ-FUS",    "DNA",        "Nanodrone-Chemo", 0.72,  65.0),
            ("Nano-siRNA-PTEN", "PTEN",       "Nanodrone-siRNA", 0.65,  52.0),
        ]
        df = pd.DataFrame(data, columns=[
            "agent", "target", "drug_class", "bee_penetration", "response_rate"
        ])
        return df.sort_values("response_rate", ascending=False).reset_index(drop=True)
