"""
modules/patient_omics.py
=========================
Integrazione dati paziente-specifici: scRNA-seq e WES (Whole Exome Sequencing).

Pipeline:
  WES   → varianti somatiche → overlap con pathway GBM → profilo mutazionale paziente
  scRNA → cluster cellulari  → firma espressione per cluster → eterogeneità tumorale
  Merge → PatientProfile      → score target personalizzato

Formato input supportato:
  WES : VCF-like (gene, chr, pos, ref, alt, VAF, depth)
  scRNA: matrice gene × cellula (sparsa, z-score normalizzata)

In assenza di dati reali → genera paziente sintetico coerente con distribuzione TCGA-GBM.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

# ── GBM driver genes (TCGA-GBM, Brennan 2013) ────────────────────────────────
GBM_DRIVERS = {
    "EGFR":   {"pathway": "RTK/RAS/PI3K",        "druggable": True,  "base_freq": 0.574},
    "PTEN":   {"pathway": "RTK/RAS/PI3K",        "druggable": False, "base_freq": 0.410},
    "TERT":   {"pathway": "Telomere/Epigenetics", "druggable": False, "base_freq": 0.720},
    "CDKN2A": {"pathway": "Cell Cycle/p53",       "druggable": False, "base_freq": 0.590},
    "TP53":   {"pathway": "Cell Cycle/p53",       "druggable": False, "base_freq": 0.310},
    "NF1":    {"pathway": "RTK/RAS/PI3K",         "druggable": False, "base_freq": 0.100},
    "PIK3CA": {"pathway": "RTK/RAS/PI3K",         "druggable": True,  "base_freq": 0.154},
    "CDK4":   {"pathway": "Cell Cycle/p53",       "druggable": True,  "base_freq": 0.080},
    "MDM2":   {"pathway": "Cell Cycle/p53",       "druggable": True,  "base_freq": 0.070},
    "VEGFA":  {"pathway": "Angiogenesis",         "druggable": True,  "base_freq": 0.250},
    "RB1":    {"pathway": "Cell Cycle/p53",       "druggable": False, "base_freq": 0.110},
    "PDGFRA": {"pathway": "RTK/RAS/PI3K",         "druggable": True,  "base_freq": 0.088},
    "MET":    {"pathway": "RTK/RAS/PI3K",         "druggable": True,  "base_freq": 0.040},
    "PIK3R1": {"pathway": "RTK/RAS/PI3K",         "druggable": True,  "base_freq": 0.108},
    "ATRX":   {"pathway": "Chromatin",            "druggable": False, "base_freq": 0.070},
}

# ── scRNA cluster signatures ──────────────────────────────────────────────────
SCRNA_CLUSTERS = {
    "Mesenchymal":    {"markers": ["CD44", "CHI3L1", "VIM", "FN1"],    "malignancy": 0.92},
    "Proneural":      {"markers": ["SOX2", "OLIG2", "PDGFRA", "NKX2"], "malignancy": 0.78},
    "Classical":      {"markers": ["EGFR", "PTEN", "CDK4", "CDKN2A"], "malignancy": 0.85},
    "Stem-like":      {"markers": ["NESTIN", "SOX2", "CD133", "BMI1"], "malignancy": 0.95},
    "Inflammatory":   {"markers": ["CD68", "IBA1", "CCL2", "IL6"],    "malignancy": 0.20},
    "Oligodendrocyte":{"markers": ["MBP", "CNP", "MOG", "PLP1"],      "malignancy": 0.05},
}


@dataclass
class WESVariant:
    gene:    str
    chrom:   str
    pos:     int
    ref:     str
    alt:     str
    vaf:     float     # Variant Allele Frequency [0-1]
    depth:   int       # read depth
    effect:  str = "missense"


@dataclass
class PatientProfile:
    patient_id:      str
    age:             int
    idh_status:      str          # wildtype | mutant
    mgmt_status:     str          # methylated | unmethylated
    mutations:       List[WESVariant] = field(default_factory=list)
    cluster_fractions: Dict[str, float] = field(default_factory=dict)
    top_targets:     List[str] = field(default_factory=list)
    heterogeneity_score: float = 0.0
    personalized_score: Dict[str, float] = field(default_factory=dict)


class PatientOmicsLoader:
    """
    Carica e processa dati omici paziente-specifici per GBM.

    Args:
        wes_path  : Path a file VCF/TSV (opzionale; se None genera dati sintetici)
        scrna_path: Path a matrice scRNA (opzionale; se None genera dati sintetici)
        seed      : seed per riproducibilità dati sintetici
    """

    def __init__(
        self,
        wes_path:   Optional[Path] = None,
        scrna_path: Optional[Path] = None,
        seed: int = 42,
    ):
        self.wes_path   = wes_path
        self.scrna_path = scrna_path
        self.rng        = np.random.default_rng(seed)

    # ── WES ────────────────────────────────────────────────────────────────
    def load_wes(self, patient_id: str = "PT-001") -> pd.DataFrame:
        """
        Carica varianti WES o genera profilo sintetico coerente con TCGA-GBM.
        Returns: DataFrame con colonne gene, chrom, pos, ref, alt, vaf, depth, effect
        """
        if self.wes_path and Path(self.wes_path).exists():
            return self._parse_vcf(self.wes_path)
        return self._synthetic_wes(patient_id)

    def _parse_vcf(self, path: Path) -> pd.DataFrame:
        """Parser semplificato per VCF/TSV con header."""
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    rows.append({
                        "gene":  parts[0],  "chrom": parts[1],
                        "pos":   int(parts[2]) if parts[2].isdigit() else 0,
                        "ref":   parts[3],  "alt":   parts[4],
                        "vaf":   float(parts[5]) if len(parts) > 5 else 0.5,
                        "depth": int(parts[6])   if len(parts) > 6 else 100,
                        "effect": parts[7]       if len(parts) > 7 else "missense",
                    })
        return pd.DataFrame(rows)

    def _synthetic_wes(self, patient_id: str) -> pd.DataFrame:
        """
        Genera profilo mutazionale sintetico campionando dalle frequenze TCGA-GBM.
        Ogni paziente ha un subset unico di mutazioni driver.
        """
        rows = []
        chromosomes = {"EGFR":"7","PTEN":"10","TERT":"5","CDKN2A":"9",
                       "TP53":"17","NF1":"17","PIK3CA":"3","CDK4":"12",
                       "MDM2":"12","VEGFA":"6","RB1":"13","PDGFRA":"4",
                       "MET":"7","PIK3R1":"5","ATRX":"X"}
        effects = ["amplification","deletion","missense","frameshift","promoter_mutation","splice_site"]

        for gene, info in GBM_DRIVERS.items():
            if self.rng.random() < info["base_freq"]:
                vaf   = float(self.rng.uniform(0.15, 0.85))
                depth = int(self.rng.integers(80, 400))
                rows.append(WESVariant(
                    gene=gene,
                    chrom=chromosomes.get(gene, str(self.rng.integers(1, 23))),
                    pos=int(self.rng.integers(10_000_000, 200_000_000)),
                    ref=self.rng.choice(list("ACGT")),
                    alt=self.rng.choice(list("ACGT")),
                    vaf=round(vaf, 3),
                    depth=depth,
                    effect=str(self.rng.choice(effects)),
                ).__dict__)

        # Aggiungi sempre TERT (driver quasi universale in IDH-wt)
        if not any(r["gene"] == "TERT" for r in rows):
            rows.append(WESVariant("TERT","5",1295228,"C","T",
                                   0.72, 250, "promoter_mutation").__dict__)

        df = pd.DataFrame(rows)
        df["patient_id"] = patient_id
        return df.sort_values("vaf", ascending=False).reset_index(drop=True)

    # ── scRNA-seq ──────────────────────────────────────────────────────────
    def load_scrna(self, n_cells: int = 500) -> pd.DataFrame:
        """
        Carica/genera matrice scRNA-seq.
        Returns: DataFrame cell × [cluster, malignancy, marker_scores...]
        """
        if self.scrna_path and Path(self.scrna_path).exists():
            return pd.read_csv(self.scrna_path)
        return self._synthetic_scrna(n_cells)

    def _synthetic_scrna(self, n_cells: int) -> pd.DataFrame:
        """
        Genera matrice scRNA sintetica con distribuzione realistica dei cluster GBM.
        Basata su: Neftel et al. Cell 2019 (GBM single-cell atlas).
        """
        # Proporzioni tipiche in GBM IDH-wt (Neftel 2019)
        cluster_props = {
            "Mesenchymal":    0.28,
            "Classical":      0.22,
            "Proneural":      0.15,
            "Stem-like":      0.18,
            "Inflammatory":   0.12,
            "Oligodendrocyte":0.05,
        }
        rows = []
        for cluster, prop in cluster_props.items():
            n = max(1, int(n_cells * prop))
            sig = SCRNA_CLUSTERS[cluster]
            for i in range(n):
                noise = self.rng.normal(0, 0.15)
                row = {
                    "cell_id":          f"{cluster[:3].upper()}{i:04d}",
                    "cluster":          cluster,
                    "malignancy_score": min(1.0, max(0.0, sig["malignancy"] + noise)),
                    "n_genes_detected": int(self.rng.integers(800, 4500)),
                    "total_counts":     int(self.rng.integers(2000, 25000)),
                    "mito_fraction":    round(float(self.rng.uniform(0.01, 0.25)), 3),
                }
                # Score per marker principale
                for marker in sig["markers"]:
                    row[f"expr_{marker}"] = max(0.0, float(
                        self.rng.normal(2.5, 0.8) if self.rng.random() < 0.7 else
                        self.rng.normal(0.2, 0.3)
                    ))
                rows.append(row)

        df = pd.DataFrame(rows)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Integrazione → PatientProfile ────────────────────────────────────
    def build_patient_profile(
        self,
        patient_id:   str = "PT-001",
        age:          int = 58,
        idh_status:   str = "wildtype",
        mgmt_status:  str = "unmethylated",
    ) -> PatientProfile:
        """
        Costruisce profilo paziente completo da WES + scRNA.
        """
        wes_df   = self.load_wes(patient_id)
        scrna_df = self.load_scrna()

        # Cluster fractions
        cluster_counts = scrna_df["cluster"].value_counts(normalize=True).to_dict()

        # Eterogeneità: Shannon entropy sui cluster maligni
        malignant_props = {
            k: v for k, v in cluster_counts.items()
            if SCRNA_CLUSTERS[k]["malignancy"] > 0.5
        }
        total = sum(malignant_props.values()) + 1e-9
        entropy = -sum((p/total) * np.log(p/total + 1e-9)
                       for p in malignant_props.values())
        heterogeneity = min(1.0, entropy / np.log(len(malignant_props) + 1))

        # Score personalizzato per ogni gene mutato
        personalized = {}
        for _, row in wes_df.iterrows():
            gene = row["gene"]
            info = GBM_DRIVERS.get(gene, {})
            # Score = VAF × druggability_bonus × cluster_enrichment
            druggable_bonus = 1.4 if info.get("druggable") else 1.0
            cluster_enrich  = max(1.0, cluster_counts.get("Classical", 0) * 3 +
                                  cluster_counts.get("Mesenchymal", 0) * 2)
            score = round(row["vaf"] * druggable_bonus * cluster_enrich, 4)
            personalized[gene] = score

        top_targets = sorted(personalized, key=personalized.get, reverse=True)[:5]

        # Converti wes_df rows in WESVariant
        mutations = [
            WESVariant(**{k: v for k, v in r.items()
                          if k in WESVariant.__dataclass_fields__})
            for _, r in wes_df.iterrows()
        ]

        return PatientProfile(
            patient_id=patient_id,
            age=age,
            idh_status=idh_status,
            mgmt_status=mgmt_status,
            mutations=mutations,
            cluster_fractions=cluster_counts,
            top_targets=top_targets,
            heterogeneity_score=round(heterogeneity, 4),
            personalized_score=personalized,
        )

    # ── Report summary ──────────────────────────────────────────────────────
    def profile_summary(self, profile: PatientProfile) -> str:
        lines = [
            f"Patient: {profile.patient_id}  Age: {profile.age}",
            f"IDH: {profile.idh_status}  MGMT: {profile.mgmt_status}",
            f"Mutations detected: {len(profile.mutations)}",
            f"Heterogeneity score: {profile.heterogeneity_score:.3f}",
            f"Top targets: {', '.join(profile.top_targets)}",
            f"Dominant cluster: {max(profile.cluster_fractions, key=profile.cluster_fractions.get)}"
            if profile.cluster_fractions else "",
        ]
        return "\n".join(lines)
