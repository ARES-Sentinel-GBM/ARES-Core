"""
modules/rl_optimizer.py
========================
Agente Q-Learning per ottimizzazione configurazione flotta nanodrone in GBM.

Spazio degli stati: (dose_level, route_idx, coating_idx, fleet_ratio_idx)
Spazio delle azioni: modifiche ai parametri della flotta
Reward:  alpha * bee_penetration + beta * peak_effect - gamma * systemic_toxicity
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute, PKParameters
)


# ── Spazio discreto dei parametri ────────────────────────────────────────────
DOSE_LEVELS   = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]          # a.u.
ROUTES        = list(DeliveryRoute)
COATINGS      = ["PEG", "Transferrin", "PEG+Transferrin", "Angiopep2", "Lactoferrin"]
FLEET_RATIOS  = [
    (4, 1, 2, 3),   # conservative
    (6, 1, 3, 3),   # standard (configurazione originale)
    (4, 2, 4, 2),   # aggressive attack
    (3, 1, 5, 4),   # max payload
    (6, 2, 3, 2),   # balanced
]

N_DOSES   = len(DOSE_LEVELS)
N_ROUTES  = len(ROUTES)
N_COAT    = len(COATINGS)
N_FLEET   = len(FLEET_RATIOS)
N_STATES  = N_DOSES * N_ROUTES * N_COAT * N_FLEET

# Reward weights
ALPHA = 0.50   # BEE penetration
BETA  = 0.35   # peak effect
GAMMA = 0.15   # penalità tossicità sistemica


# ── Encoding stato → indice ───────────────────────────────────────────────────
def encode_state(dose_i: int, route_i: int, coat_i: int, fleet_i: int) -> int:
    return dose_i * (N_ROUTES * N_COAT * N_FLEET) + \
           route_i * (N_COAT * N_FLEET) + \
           coat_i * N_FLEET + fleet_i

def decode_state(state: int) -> Tuple[int, int, int, int]:
    fleet_i = state % N_FLEET
    state  //= N_FLEET
    coat_i  = state % N_COAT
    state  //= N_COAT
    route_i = state % N_ROUTES
    dose_i  = state // N_ROUTES
    return dose_i, route_i, coat_i, fleet_i


# ── Ambiente GBM ─────────────────────────────────────────────────────────────
class GBMEnvironment:
    """
    Ambiente che valuta una configurazione nanodrone e restituisce reward.
    Simulazione abbreviata (24h) per velocità nel training RL.
    """

    def __init__(self, target_gene: str = "EGFR", duration_h: float = 24.0):
        self.target_gene = target_gene
        self.duration_h  = duration_h
        self.pk_params   = PKParameters(bee_base=0.05, tumor_icp=25.0, tumor_ifp=22.0)

    def evaluate(
        self,
        dose_i: int, route_i: int, coat_i: int, fleet_i: int
    ) -> Tuple[float, dict]:
        """
        Valuta la configurazione e restituisce (reward, info_dict).
        """
        dose    = DOSE_LEVELS[dose_i]
        route   = ROUTES[route_i]
        coating = COATINGS[coat_i]
        s, d, a, l = FLEET_RATIOS[fleet_i]

        spec = DroneSpec(
            size_nm=120,
            surface_coating=coating,
            payload_type="chemo",
            immune_evasion=0.72,
            targeting_affinity=0.65,
        )
        fleet = FleetConfig(
            n_sentinelle=s, n_decisori=d, n_attacco=a, n_lager=l,
            delivery_route=route, drone_spec=spec,
        )
        sim = NanodronePKSimulator(fleet, self.pk_params)

        try:
            pk_df   = sim.simulate(dose=dose, duration_h=self.duration_h)
            summary = sim.pk_summary(pk_df)
        except Exception:
            return -1.0, {}

        bee    = summary["bee_penetration"]
        effect = summary["peak_effect"]

        # Tossicità sistemica proxy: alta dose + bassa selettività route = più tossico
        tox = (dose / max(DOSE_LEVELS)) * (1.0 - bee * 0.5)

        reward = ALPHA * bee + BETA * effect - GAMMA * tox

        return round(reward, 5), {
            "dose":           dose,
            "route":          route.value,
            "coating":        coating,
            "fleet":          (s, d, a, l),
            "bee":            bee,
            "peak_effect":    effect,
            "toxicity_proxy": tox,
            "reward":         reward,
        }


# ── Agente Q-Learning ────────────────────────────────────────────────────────
@dataclass
class QLearningConfig:
    episodes:     int   = 600
    alpha:        float = 0.15    # learning rate
    gamma_rl:     float = 0.90    # discount factor
    epsilon_start: float = 1.00   # esplorazione iniziale
    epsilon_end:  float  = 0.05   # esplorazione minima
    epsilon_decay: float = 0.992  # decadimento per episodio


class NanodronQAgent:
    """
    Agente Q-Learning tabellare per ottimizzazione configurazione nanodrone.

    Ogni episodio:
      1. Campiona stato iniziale casuale
      2. Seleziona azione ε-greedy
      3. Valuta nell'ambiente GBM
      4. Aggiorna Q-table
      5. Traccia reward cumulativo
    """

    def __init__(self, config: QLearningConfig = None):
        self.cfg     = config or QLearningConfig()
        self.n_act   = N_STATES   # ogni stato è un'azione (selezione diretta configurazione)
        self.Q       = np.zeros(N_STATES)    # Q-table 1D (stateless: ogni config è un'azione)
        self.env     = GBMEnvironment()
        self.history: list[dict] = []

    def _epsilon(self, episode: int) -> float:
        eps = self.cfg.epsilon_start * (self.cfg.epsilon_decay ** episode)
        return max(self.cfg.epsilon_end, eps)

    def _sample_action(self, episode: int) -> int:
        if np.random.random() < self._epsilon(episode):
            return np.random.randint(N_STATES)
        return int(np.argmax(self.Q))

    def train(self) -> pd.DataFrame:
        """
        Esegue il training completo. Restituisce history DataFrame.
        """
        for ep in range(self.cfg.episodes):
            action = self._sample_action(ep)
            dose_i, route_i, coat_i, fleet_i = decode_state(action)

            reward, info = self.env.evaluate(dose_i, route_i, coat_i, fleet_i)

            # Update Q-table (bandit semplificato: no transizioni di stato)
            self.Q[action] += self.cfg.alpha * (reward - self.Q[action])

            self.history.append({
                "episode":      ep + 1,
                "action":       action,
                "reward":       reward,
                "epsilon":      self._epsilon(ep),
                "q_max":        float(np.max(self.Q)),
                "bee":          info.get("bee", 0),
                "peak_effect":  info.get("peak_effect", 0),
                "route":        info.get("route", ""),
                "coating":      info.get("coating", ""),
            })

        return pd.DataFrame(self.history)

    def best_config(self) -> dict:
        """Restituisce la configurazione con Q-value massimo."""
        best_action = int(np.argmax(self.Q))
        dose_i, route_i, coat_i, fleet_i = decode_state(best_action)
        _, info = self.env.evaluate(dose_i, route_i, coat_i, fleet_i)
        info["q_value"]    = float(self.Q[best_action])
        info["action_idx"] = best_action
        return info

    def top_configs(self, n: int = 5) -> pd.DataFrame:
        """Restituisce le top-n configurazioni per Q-value."""
        top_idx = np.argsort(self.Q)[::-1][:n]
        rows = []
        for idx in top_idx:
            dose_i, route_i, coat_i, fleet_i = decode_state(int(idx))
            _, info = self.env.evaluate(dose_i, route_i, coat_i, fleet_i)
            info["q_value"] = float(self.Q[idx])
            info["rank"]    = len(rows) + 1
            rows.append(info)
        return pd.DataFrame(rows)
