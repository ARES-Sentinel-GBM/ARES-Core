"""
modules/pk3d_mesh.py
=====================
Modello PK 3D su mesh volumetrica tumorale.

Approccio: diffusione di Fick 3D + convection su griglia cartesiana (FDM).
Equazione governante:
    ∂C/∂t = D·∇²C − v·∇C − k_el·C + S(t)

Dove:
    C   = concentrazione nanodrone [a.u./mm³]
    D   = coefficiente di diffusione efficace nel tessuto GBM
    v   = velocità convettiva (pressione IFP)
    k_el= rate di eliminazione locale
    S   = sorgente (iniezione/infusione)

La mesh è una geometria ellissoidale (approssimazione forma tumore GBM).
Bordi = condizione di Dirichlet (C=0 alla periferia BEE).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TumorGeometry:
    """Geometria ellissoidale del tumore GBM."""
    # Semiassi [mm] — GBM tipico: volume ~30-60 cm³
    rx: float = 22.0   # asse x
    ry: float = 18.0   # asse y
    rz: float = 16.0   # asse z
    # Risoluzione griglia
    nx: int = 30
    ny: int = 26
    nz: int = 24
    # Centro iniezione relativo (% degli assi)
    src_x: float = 0.0
    src_y: float = 0.0
    src_z: float = 0.0


@dataclass
class TissuePKParams:
    """Parametri fisici del tessuto GBM per la diffusione."""
    D_eff:   float = 0.08    # cm²/h  diffusione efficace in tessuto GBM
    k_el:    float = 0.06    # h⁻¹   eliminazione locale (degradazione, uptake)
    v_conv:  float = 0.02    # cm/h   velocità convettiva (IFP-driven)
    bee_frac: float = 0.72   # BEE penetration (modula S alla sorgente)
    ifp_mmhg: float = 22.0   # Pressione interstiziale [mmHg]

    def effective_diffusion(self) -> float:
        """D ridotto da IFP elevata (compressione spazio extracellulare)."""
        ifp_penalty = max(0.5, 1.0 - (self.ifp_mmhg - 10.0) * 0.01)
        return self.D_eff * ifp_penalty


class PK3DMeshSimulator:
    """
    Simulatore PK 3D su mesh tumorale GBM.

    Utilizza differenze finite (Eulero esplicito) per risolvere
    l'equazione di diffusione-convection-reaction sulla geometria ellissoidale.

    Args:
        geometry : TumorGeometry con dimensioni e risoluzione
        pk       : TissuePKParams con parametri fisici
    """

    def __init__(
        self,
        geometry: TumorGeometry = None,
        pk:       TissuePKParams = None,
    ):
        self.geo = geometry or TumorGeometry()
        self.pk  = pk       or TissuePKParams()
        self._build_mesh()

    def _build_mesh(self):
        """Costruisce griglia cartesiana e maschera ellissoidale."""
        g = self.geo
        # Griglie coordinate [mm]
        self.x = np.linspace(-g.rx, g.rx, g.nx)
        self.y = np.linspace(-g.ry, g.ry, g.ny)
        self.z = np.linspace(-g.rz, g.rz, g.nz)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        # Maschera ellissoidale: punti interni al tumore
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.mask = ((X/g.rx)**2 + (Y/g.ry)**2 + (Z/g.rz)**2) <= 1.0
        self.tumor_voxels = int(np.sum(self.mask))
        self.tumor_volume_mm3 = self.tumor_voxels * self.dx * self.dy * self.dz

        # Indice della sorgente (centro o posizione specificata)
        self._src_i = int((g.src_x + g.rx) / (2*g.rx) * (g.nx - 1))
        self._src_j = int((g.src_y + g.ry) / (2*g.ry) * (g.ny - 1))
        self._src_k = int((g.src_z + g.rz) / (2*g.rz) * (g.nz - 1))

    def simulate(
        self,
        dose:       float = 1.0,
        duration_h: float = 24.0,
        dt:         float = 0.10,
        target_gene: str = "EGFR",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Esegue la simulazione 3D.

        Returns:
            time_df  : DataFrame (time_h, c_mean, c_max, c_core, volume_covered_pct)
            final_map: Array 3D concentrazione al tempo finale [nx, ny, nz]
        """
        D    = self.pk.effective_diffusion()
        k_el = self.pk.k_el
        v    = self.pk.v_conv
        bee  = self.pk.bee_frac

        # CFL stability check — limita dt se necessario
        cfl_dt = 0.4 * min(self.dx, self.dy, self.dz)**2 / D
        dt = min(dt, cfl_dt)

        steps  = int(duration_h / dt)
        n_out  = min(200, steps)
        stride = max(1, steps // n_out)

        C = np.zeros((self.geo.nx, self.geo.ny, self.geo.nz))

        # Sorgente gaussiana centrata: simula iniezione/infusione
        sigma = min(self.geo.rx, self.geo.ry, self.geo.rz) * 0.25
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        src_profile = np.exp(
            -((X - self.geo.src_x)**2 + (Y - self.geo.src_y)**2 +
              (Z - self.geo.src_z)**2) / (2 * sigma**2)
        ) * self.mask
        src_profile /= (src_profile.sum() + 1e-12)

        # Infusione continua nelle prime 4 ore (simula rilascio dalla flotta)
        t_infuse = min(4.0, duration_h * 0.15)
        src_rate = dose * bee / t_infuse  # dose/h durante infusione

        records = []
        times   = np.arange(0, duration_h + dt, dt)

        for step, t in enumerate(times[:-1]):
            # Sorgente attiva nelle prime t_infuse ore
            if t < t_infuse:
                C += src_profile * src_rate * dt

            # Laplaciano (diffusione) con differenze finite centrali
            lap = (
                np.roll(C, -1, 0) + np.roll(C, 1, 0) - 2*C) / self.dx**2 + (
                np.roll(C, -1, 1) + np.roll(C, 1, 1) - 2*C) / self.dy**2 + (
                np.roll(C, -1, 2) + np.roll(C, 1, 2) - 2*C) / self.dz**2

            # Gradiente (convection upwind semplificato)
            grad_x = (np.roll(C, -1, 0) - C) / self.dx
            grad_y = (np.roll(C, -1, 1) - C) / self.dy
            grad_z = (np.roll(C, -1, 2) - C) / self.dz

            dC = (D * lap - v * (grad_x + grad_y + grad_z) - k_el * C) * dt
            C  = np.maximum(0.0, C + dC)

            # Condizioni al contorno: C=0 fuori dal tumore
            C[~self.mask] = 0.0

            # Registra output
            if step % stride == 0:
                C_tumor = C[self.mask]
                c_mean  = float(C_tumor.mean()) if C_tumor.size > 0 else 0.0
                c_max   = float(C_tumor.max())  if C_tumor.size > 0 else 0.0
                # Concentrazione nel core tumorale (voxel centrale)
                ci, cj, ck = self._src_i, self._src_j, self._src_k
                c_core = float(C[ci, cj, ck])
                # Volume coperto (C > 10% Cmax)
                threshold = c_max * 0.10
                vol_pct = float(np.sum(C_tumor > threshold) / (self.tumor_voxels + 1e-9) * 100)

                records.append({
                    "time_h":          round(t, 3),
                    "c_mean":          round(c_mean, 6),
                    "c_max":           round(c_max,  6),
                    "c_core":          round(c_core, 6),
                    "volume_covered_pct": round(vol_pct, 2),
                    "target_gene":     target_gene,
                })

        return pd.DataFrame(records), C

    def summary_3d(self, time_df: pd.DataFrame, final_map: np.ndarray) -> dict:
        """Riassume parametri chiave della simulazione 3D."""
        cmax_idx = time_df["c_max"].idxmax()
        return {
            "tumor_volume_mm3":    round(self.tumor_volume_mm3, 1),
            "tumor_voxels":        self.tumor_voxels,
            "grid_shape":          (self.geo.nx, self.geo.ny, self.geo.nz),
            "cmax_mean":           round(time_df["c_max"].max(), 6),
            "tmax_h":              round(time_df.iloc[cmax_idx]["time_h"], 1),
            "final_coverage_pct":  round(time_df["volume_covered_pct"].iloc[-1], 1),
            "peak_coverage_pct":   round(time_df["volume_covered_pct"].max(), 1),
            "effective_D":         round(self.pk.effective_diffusion(), 4),
            "bee_fraction":        self.pk.bee_frac,
            "ifp_mmhg":            self.pk.ifp_mmhg,
            "final_map_nonzero":   int(np.sum(final_map > 1e-8)),
        }

    def axial_slice(self, volume: np.ndarray, axis: int = 2) -> np.ndarray:
        """Ritorna slice centrale lungo l'asse specificato (0=x, 1=y, 2=z)."""
        mid = volume.shape[axis] // 2
        return np.take(volume, mid, axis=axis)
