"""
modules/nanodrone_sim.py
========================
Simulatore Farmacocinetico per flotta nanodrone in GBM.

Modello bicompartimentale plasma/GBM con:
  - Effetto ICP (pressione intracranica) e IFP (pressione interstiziale)
  - BEE penetrazione modulata da coating + metodo di delivery
  - Targeting affinity per recettore tumorale
  - Fleet orchestration: Sentinelle, Decisori, Attacco, Lager (SDAL config)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ── Delivery Routes ─────────────────────────────────────────────────────────
class DeliveryRoute(Enum):
    IV_FREE      = "IV-free"        # endovena, farmaco libero
    LIPOSOME     = "Liposome"       # liposomi PEGylati
    NP_PLGA      = "NP-PLGA"        # nanoparticelle PLGA
    TRANSFERRIN  = "Transferrin-NP" # NP + recettore transferrina (RMT)
    EXOSOME      = "Exosome"        # esosomi ingegnerizzati
    FUS          = "FUS-NP"         # Focused Ultrasound + NP (ottimale)
    CED          = "CED"            # Convection-Enhanced Delivery


# BEE penetration fraction per route
_ROUTE_BEE: dict[DeliveryRoute, float] = {
    DeliveryRoute.IV_FREE:     0.02,
    DeliveryRoute.LIPOSOME:    0.12,
    DeliveryRoute.NP_PLGA:     0.25,
    DeliveryRoute.TRANSFERRIN: 0.45,
    DeliveryRoute.EXOSOME:     0.35,
    DeliveryRoute.FUS:         0.68,
    DeliveryRoute.CED:         0.90,
}

# ── Drone Specification ──────────────────────────────────────────────────────
@dataclass
class DroneSpec:
    size_nm:           float = 120.0         # diametro nanoparticella [nm]
    surface_coating:   str   = "PEG"         # coating superficie
    payload_type:      str   = "chemo"       # chemo | siRNA | antibody
    immune_evasion:    float = 0.50          # evasione sistema immunitario [0-1]
    targeting_affinity: float = 0.55         # affinità recettore target [0-1]

    def coating_bee_bonus(self) -> float:
        """Bonus BEE aggiuntivo in base al coating."""
        bonuses = {
            "PEG":                   0.00,
            "Transferrin":           0.10,
            "PEG+Transferrin":       0.12,
            "Angiopep2":             0.08,
            "PEG+Angiopep2":         0.09,
            "Lactoferrin":           0.07,
            "PEG+Lactoferrin":       0.08,
        }
        return bonuses.get(self.surface_coating, 0.0)

# ── Fleet Configuration ──────────────────────────────────────────────────────
@dataclass
class FleetConfig:
    """
    Configurazione flotta SDAL:
      S = Sentinelle (ricognizione, monitoraggio pH/segnali)
      D = Decisori   (elaborazione, switching payload)
      A = Attacco    (rilascio payload chemioterapico)
      L = Lager      (riserva payload, rilascio prolungato)
    """
    n_sentinelle:    int          = 4
    n_decisori:      int          = 1
    n_attacco:       int          = 3
    n_lager:         int          = 2
    delivery_route:  DeliveryRoute = DeliveryRoute.FUS
    drone_spec:      DroneSpec    = field(default_factory=DroneSpec)

    @property
    def total_drones(self) -> int:
        return self.n_sentinelle + self.n_decisori + self.n_attacco + self.n_lager

    def attack_fraction(self) -> float:
        """Frazione droni attivi sul totale."""
        return (self.n_attacco + self.n_lager * 0.5) / self.total_drones

    def effective_bee(self) -> float:
        """BEE penetrazione effettiva = route_base + coating_bonus, cap 0.95."""
        base   = _ROUTE_BEE.get(self.delivery_route, 0.05)
        bonus  = self.drone_spec.coating_bee_bonus()
        return min(0.95, base + bonus)

# ── PK Parameters ────────────────────────────────────────────────────────────
@dataclass
class PKParameters:
    """Parametri fisiologici del compartimento GBM."""
    bee_base:    float = 0.05    # permeabilità BEE basale GBM (alta vs normale 0.01)
    tumor_icp:   float = 20.0   # pressione intracranica [mmHg] (normale ≤15)
    tumor_ifp:   float = 15.0   # pressione interstiziale tumorale [mmHg]
    plasma_t_half: float = 6.0  # emivita plasmatica [h]
    volume_dist:   float = 0.25  # volume distribuzione approssimato [L/kg]

# ── PK Simulator ─────────────────────────────────────────────────────────────
class NanodronePKSimulator:
    """
    Simulatore bicompartimentale plasma → GBM.

    Modello:
      dC_plasma/dt = -k_el * C_plasma - k_trans * C_plasma
      dC_gbm/dt   = k_trans * C_plasma * bee_eff * targeting_eff - k_out * C_gbm

    Pressione ICP/IFP riduce il trasporto convettivo (fattore p_penalty).
    """

    def __init__(self, fleet: FleetConfig, pk: PKParameters):
        self.fleet = fleet
        self.pk    = pk

    def _pressure_penalty(self) -> float:
        """Riduzione trasporto per alta pressione ICP/IFP."""
        icp_penalty = max(0.3, 1.0 - (self.pk.tumor_icp - 15.0) * 0.02)
        ifp_penalty = max(0.4, 1.0 - (self.pk.tumor_ifp - 10.0) * 0.015)
        return icp_penalty * ifp_penalty

    def simulate(
        self,
        dose: float = 1.0,
        duration_h: float = 72.0,
        dt: float = 0.25,
        target_gene: str = "EGFR",
    ) -> pd.DataFrame:
        """
        Simula profilo PK nel tempo per plasma e compartimento GBM.

        Returns:
            DataFrame con colonne: time_h, c_plasma, c_gbm, effect, active_drones
        """
        bee_eff      = self.fleet.effective_bee()
        attack_frac  = self.fleet.attack_fraction()
        targeting    = self.fleet.drone_spec.targeting_affinity
        immune_ev    = self.fleet.drone_spec.immune_evasion
        p_penalty    = self._pressure_penalty()

        # Rate constants [h⁻¹]
        k_el    = np.log(2) / self.pk.plasma_t_half          # eliminazione plasmatica
        k_trans = 0.15 * bee_eff * p_penalty                  # trasferimento BEE
        k_out   = 0.08                                         # clearance GBM
        k_eff   = 0.25 * targeting * attack_frac * immune_ev  # efficacia biologica

        times    = np.arange(0, duration_h + dt, dt)
        c_plasma = np.zeros_like(times)
        c_gbm    = np.zeros_like(times)
        effect   = np.zeros_like(times)
        active   = np.zeros_like(times)

        c_plasma[0] = dose

        for i in range(1, len(times)):
            dc_plasma = -(k_el + k_trans) * c_plasma[i-1]
            dc_gbm    = k_trans * c_plasma[i-1] - k_out * c_gbm[i-1]

            c_plasma[i] = max(0.0, c_plasma[i-1] + dc_plasma * dt)
            c_gbm[i]    = max(0.0, c_gbm[i-1]    + dc_gbm    * dt)
            effect[i]   = min(1.0, k_eff * c_gbm[i])

            # Droni attivi: degrado esponenziale lento
            active[i] = self.fleet.total_drones * np.exp(-0.01 * times[i])

        return pd.DataFrame({
            "time_h":       times,
            "c_plasma":     c_plasma,
            "c_gbm":        c_gbm,
            "effect":       effect,
            "active_drones": active,
            "target_gene":  target_gene,
        })

    def pk_summary(self, pk_df: pd.DataFrame) -> dict:
        """Estrae parametri PK chiave dalla simulazione."""
        cmax_idx = pk_df["c_gbm"].idxmax()
        auc_gbm  = np.trapezoid(pk_df["c_gbm"], pk_df["time_h"])

        # T½ stimato: tempo dal Cmax per scendere al 50%
        cmax_val = pk_df["c_gbm"].iloc[cmax_idx]
        half_val = cmax_val * 0.5
        post_max = pk_df.iloc[cmax_idx:]
        t_half_rows = post_max[post_max["c_gbm"] <= half_val]
        t_half = round(t_half_rows.iloc[0]["time_h"] - pk_df.iloc[cmax_idx]["time_h"], 1) \
            if not t_half_rows.empty else ">72h"

        return {
            "bee_penetration":  round(self.fleet.effective_bee(), 4),
            "cmax_gbm":         round(cmax_val, 4),
            "tmax_h":           round(pk_df.iloc[cmax_idx]["time_h"], 1),
            "t_half_h":         t_half,
            "auc_gbm":          round(auc_gbm, 4),
            "peak_effect":      round(pk_df["effect"].max(), 4),
            "total_drones":     self.fleet.total_drones,
            "delivery_route":   self.fleet.delivery_route.value,
        }


# ── Drug Comparison ───────────────────────────────────────────────────────────
class DrugComparison:
    """
    Comparazione nanodroni vs farmaci convenzionali per GBM.
    Efficacy score composito: BEE * selectivity * response_potential.
    """

    _AGENTS = [
        # agent               bee    selectivity  resp_potential  is_nano
        ("Nano-TMZ-FUS",      0.72,  0.82,        0.90, True),
        ("Nano-EGFR-Ab",      0.68,  0.88,        0.85, True),
        ("Nano-siRNA-PTEN",   0.65,  0.85,        0.80, True),
        ("Carmustine (BCNU)",  0.55,  0.35,       0.62, False),
        ("Lomustine (CCNU)",   0.60,  0.38,       0.60, False),
        ("Temozolomide",       0.40,  0.45,        0.55, False),
        ("Osimertinib",        0.45,  0.75,        0.55, False),
        ("Palbociclib",        0.25,  0.70,        0.45, False),
        ("Erlotinib",          0.30,  0.65,        0.40, False),
        ("Bevacizumab",        0.02,  0.60,        0.35, False),
        ("Irinotecan",         0.15,  0.40,        0.32, False),
        ("Alpelisib",          0.22,  0.68,        0.38, False),
    ]

    def compare(self) -> pd.DataFrame:
        rows = []
        for agent, bee, sel, resp, is_nano in self._AGENTS:
            score = round(0.40 * bee + 0.35 * sel + 0.25 * resp, 4)
            rows.append({
                "agent":          agent,
                "bee_penetration": bee,
                "selectivity":    sel,
                "response_potential": resp,
                "efficacy_score": score,
                "is_nanodrone":   is_nano,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("efficacy_score", ascending=False).reset_index(drop=True)
