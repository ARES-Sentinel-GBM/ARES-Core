"""
modules/dqn_optimizer.py
=========================
Deep Q-Network (DQN) per ottimizzazione nanodrone con spazio d'azione continuo.

Architettura:
  Input  : stato (6 feature normalizzate)
  Hidden : 64 → 32 neuroni (ReLU)
  Output : Q-values per N_ACTIONS azioni discrete (quantizzazione dello spazio continuo)

Feature di stato:
  [bee_current, effect_current, dose_norm, icp_norm, ifp_norm, t_norm]

Azioni (spazio continuo quantizzato):
  Δ_dose, Δ_route, Δ_coating, Δ_fleet_ratio

Algoritmo: DQN con:
  - Experience Replay (ReplayBuffer)
  - Target Network (hard update ogni C step)
  - ε-greedy decrescente
  - Huber loss (robust ai outlier)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple

from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute, PKParameters
)

# ── Spazio d'azione discretizzato ─────────────────────────────────────────────
# 4 dimensioni d'azione × granularità
DOSE_VALUES    = np.linspace(0.25, 3.0, 12)    # 12 livelli dose
ROUTE_MAP      = list(DeliveryRoute)             # 7 route
COATING_MAP    = ["PEG", "Transferrin", "PEG+Transferrin",
                  "Angiopep2", "Lactoferrin"]    # 5 coating
FLEET_MAP      = [                               # 6 configurazioni fleet
    (4,1,2,2), (6,1,3,3), (4,2,4,2),
    (3,1,5,4), (6,2,3,2), (8,1,4,4),
]

# Spazio d'azione flat: tutti i prodotti cartesiani
import itertools as _it
ACTION_SPACE = list(_it.product(
    range(len(DOSE_VALUES)),
    range(len(ROUTE_MAP)),
    range(len(COATING_MAP)),
    range(len(FLEET_MAP)),
))
N_ACTIONS = len(ACTION_SPACE)   # 12×7×5×6 = 2520

N_FEATURES = 6   # dimensione stato


# ── Rete neurale (pure numpy) ──────────────────────────────────────────────────
class SimpleNN:
    """
    Rete MLP 2-layer: N_FEATURES → 64 → 32 → N_ACTIONS.
    Pesi inizializzati con He (ReLU).
    """
    def __init__(self, n_in: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((64, n_in))  * np.sqrt(2 / n_in)
        self.b1 = np.zeros(64)
        self.W2 = rng.standard_normal((32, 64))    * np.sqrt(2 / 64)
        self.b2 = np.zeros(32)
        self.W3 = rng.standard_normal((n_out, 32)) * np.sqrt(2 / 32)
        self.b3 = np.zeros(n_out)

    def _relu(self, x):  return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._relu(self.W1 @ x + self.b1)
        h2 = self._relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """X: shape (batch, n_in)"""
        out = np.zeros((len(X), self.W3.shape[0]))
        for i, x in enumerate(X):
            out[i] = self.forward(x)
        return out

    def copy_weights(self, other: "SimpleNN"):
        """Copia pesi da other → self (target network update)."""
        self.W1[:] = other.W1; self.b1[:] = other.b1
        self.W2[:] = other.W2; self.b2[:] = other.b2
        self.W3[:] = other.W3; self.b3[:] = other.b3

    def update_sgd(self, grad_W1, grad_b1, grad_W2, grad_b2,
                   grad_W3, grad_b3, lr: float = 1e-3):
        """Aggiornamento SGD con gradient clipping."""
        clip = 1.0
        self.W1 -= lr * np.clip(grad_W1, -clip, clip)
        self.b1 -= lr * np.clip(grad_b1, -clip, clip)
        self.W2 -= lr * np.clip(grad_W2, -clip, clip)
        self.b2 -= lr * np.clip(grad_b2, -clip, clip)
        self.W3 -= lr * np.clip(grad_W3, -clip, clip)
        self.b3 -= lr * np.clip(grad_b3, -clip, clip)


# ── Experience Replay ──────────────────────────────────────────────────────────
@dataclass
class Transition:
    state:      np.ndarray
    action_idx: int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self._buf = deque(maxlen=capacity)

    def push(self, t: Transition): self._buf.append(t)
    def __len__(self): return len(self._buf)

    def sample(self, batch: int, rng) -> List[Transition]:
        idx = rng.integers(0, len(self._buf), size=batch)
        return [self._buf[i] for i in idx]


# ── Ambiente DQN ───────────────────────────────────────────────────────────────
class DQNEnvironment:
    """Ambiente GBM per DQN — stato continuo 6D, reward scalare."""

    def __init__(self):
        self.pk_base = PKParameters(bee_base=0.05, tumor_icp=25.0, tumor_ifp=22.0)

    def encode_state(
        self, bee: float, effect: float, dose: float,
        icp: float, ifp: float, t_frac: float
    ) -> np.ndarray:
        return np.array([
            bee,
            effect,
            (dose - 0.25) / 2.75,
            (icp  - 10.0) / 30.0,
            (ifp  - 5.0)  / 30.0,
            t_frac,
        ], dtype=np.float32)

    def step(self, action_idx: int, prev_bee: float = 0.0) -> Tuple[np.ndarray, float, dict]:
        di, ri, ci, fi = ACTION_SPACE[action_idx]
        dose    = DOSE_VALUES[di]
        route   = ROUTE_MAP[ri]
        coating = COATING_MAP[ci]
        s, d, a, l = FLEET_MAP[fi]

        spec  = DroneSpec(surface_coating=coating, immune_evasion=0.72,
                          targeting_affinity=0.65)
        fleet = FleetConfig(n_sentinelle=s, n_decisori=d, n_attacco=a, n_lager=l,
                            delivery_route=route, drone_spec=spec)
        sim   = NanodronePKSimulator(fleet, self.pk_base)
        try:
            pk_df   = sim.simulate(dose=dose, duration_h=12.0)
            summary = sim.pk_summary(pk_df)
            bee     = summary["bee_penetration"]
            effect  = summary["peak_effect"]
            tox     = (dose / 3.0) * (1.0 - bee * 0.5)
            reward  = 0.50*bee + 0.35*effect - 0.15*tox + 0.05*max(0, bee - prev_bee)
        except Exception:
            bee = effect = 0.0
            reward = -0.5

        next_state = self.encode_state(bee, effect, dose,
                                       self.pk_base.tumor_icp,
                                       self.pk_base.tumor_ifp, 0.0)
        return next_state, round(reward, 5), {"bee": bee, "effect": effect, "dose": dose,
                                              "route": route.value, "coating": coating}


# ── DQN Agent ──────────────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    episodes:       int   = 300
    batch_size:     int   = 32
    gamma:          float = 0.92
    lr:             float = 2e-3
    epsilon_start:  float = 1.0
    epsilon_end:    float = 0.05
    epsilon_decay:  float = 0.990
    target_update:  int   = 20     # aggiorna target network ogni N episodi
    buffer_capacity: int  = 3000
    warmup_episodes: int  = 15     # episodi random prima del training


class DQNNanodrone:
    """
    Agente DQN per ottimizzazione continua della configurazione nanodrone GBM.
    Usa Experience Replay + Target Network per stabilizzare il training.
    """

    def __init__(self, config: DQNConfig = None):
        self.cfg     = config or DQNConfig()
        self.rng     = np.random.default_rng(42)
        self.env     = DQNEnvironment()
        self.net     = SimpleNN(N_FEATURES, N_ACTIONS, seed=0)
        self.target  = SimpleNN(N_FEATURES, N_ACTIONS, seed=0)
        self.target.copy_weights(self.net)
        self.buffer  = ReplayBuffer(self.cfg.buffer_capacity)
        self.history: list[dict] = []

    def _epsilon(self, ep: int) -> float:
        return max(self.cfg.epsilon_end,
                   self.cfg.epsilon_start * (self.cfg.epsilon_decay ** ep))

    def _select_action(self, state: np.ndarray, ep: int) -> int:
        if self.rng.random() < self._epsilon(ep):
            return int(self.rng.integers(N_ACTIONS))
        q = self.net.forward(state)
        return int(np.argmax(q))

    def _huber(self, x, delta=1.0):
        return np.where(np.abs(x) <= delta, 0.5*x**2, delta*(np.abs(x) - 0.5*delta))

    def _train_step(self):
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0

        batch = self.buffer.sample(self.cfg.batch_size, self.rng)
        states     = np.array([t.state      for t in batch], dtype=np.float32)
        next_states= np.array([t.next_state for t in batch], dtype=np.float32)
        actions    = np.array([t.action_idx for t in batch], dtype=np.int32)
        rewards    = np.array([t.reward     for t in batch], dtype=np.float32)
        dones      = np.array([t.done       for t in batch], dtype=np.float32)

        # Q(s,a) corrente
        Q_pred  = self.net.predict_batch(states)       # (batch, N_ACTIONS)
        # Q(s',a') target
        Q_next  = self.target.predict_batch(next_states)
        Q_target = Q_pred.copy()

        for i in range(self.cfg.batch_size):
            td = rewards[i] + (1-dones[i]) * self.cfg.gamma * np.max(Q_next[i])
            Q_target[i, actions[i]] = td

        # Backprop manuale (MSE su output layer scelto)
        loss = float(np.mean(self._huber(Q_pred - Q_target)))

        # Gradient (semplificato: solo output layer — adeguato per questo scopo)
        delta_out = (Q_pred - Q_target) / self.cfg.batch_size   # (batch, N_ACTIONS)

        # Forward pass per gradients
        for i, x in enumerate(states):
            h1 = np.maximum(0, self.net.W1 @ x + self.net.b1)
            h2 = np.maximum(0, self.net.W2 @ h1 + self.net.b2)

            d3 = delta_out[i]
            dW3 = np.outer(d3, h2)
            db3 = d3

            d2 = (self.net.W3.T @ d3) * (h2 > 0)
            dW2 = np.outer(d2, h1)
            db2 = d2

            d1 = (self.net.W2.T @ d2) * (h1 > 0)
            dW1 = np.outer(d1, x)
            db1 = d1

            self.net.update_sgd(dW1/self.cfg.batch_size, db1/self.cfg.batch_size,
                                 dW2/self.cfg.batch_size, db2/self.cfg.batch_size,
                                 dW3/self.cfg.batch_size, db3/self.cfg.batch_size,
                                 lr=self.cfg.lr)
        return loss

    def train(self) -> pd.DataFrame:
        """Esegue il training DQN completo."""
        state = self.env.encode_state(0.0, 0.0, 1.0, 25.0, 22.0, 0.0)
        prev_bee = 0.0

        for ep in range(self.cfg.episodes):
            action_idx = self._select_action(state, ep)
            next_state, reward, info = self.env.step(action_idx, prev_bee)
            done = True   # episodio singolo (bandit contestualizzato)

            self.buffer.push(Transition(state, action_idx, reward, next_state, done))

            loss = 0.0
            if ep >= self.cfg.warmup_episodes:
                loss = self._train_step()

            if ep % self.cfg.target_update == 0:
                self.target.copy_weights(self.net)

            prev_bee = info.get("bee", 0.0)
            state    = next_state

            self.history.append({
                "episode":   ep + 1,
                "reward":    reward,
                "loss":      loss,
                "epsilon":   self._epsilon(ep),
                "bee":       info.get("bee", 0.0),
                "effect":    info.get("effect", 0.0),
                "route":     info.get("route", ""),
                "coating":   info.get("coating", ""),
                "dose":      info.get("dose", 0.0),
                "q_mean":    float(np.mean(self.net.forward(state))),
            })

        return pd.DataFrame(self.history)

    def best_config(self) -> dict:
        """Valuta tutte le azioni con la Q-net finale e restituisce la migliore."""
        best_action = -1
        best_q      = -np.inf
        state = self.env.encode_state(0.5, 0.5, 1.0, 25.0, 22.0, 0.0)
        q_all = self.net.forward(state)
        best_action = int(np.argmax(q_all))

        _, reward, info = self.env.step(best_action)
        info["q_value"]    = float(np.max(q_all))
        info["reward"]     = reward
        info["action_idx"] = best_action
        return info

    def top_configs(self, n: int = 5) -> pd.DataFrame:
        state  = self.env.encode_state(0.5, 0.5, 1.0, 25.0, 22.0, 0.0)
        q_all  = self.net.forward(state)
        top_idx = np.argsort(q_all)[::-1][:n]
        rows = []
        for rank, idx in enumerate(top_idx, 1):
            _, reward, info = self.env.step(int(idx))
            info["rank"]    = rank
            info["q_value"] = float(q_all[idx])
            info["reward"]  = reward
            rows.append(info)
        return pd.DataFrame(rows)
