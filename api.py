"""
api.py — ARES-Sentinel-GBM REST API
=====================================
FastAPI REST API per integrazione con piattaforme cliniche e sistemi esterni.

Endpoints:
  GET  /health                     → status pipeline
  GET  /targets                    → top target GBM
  POST /patient/profile            → crea profilo paziente da WES/scRNA
  POST /pk/simulate                → simula PK nanodrone
  POST /pk/simulate3d              → simulazione 3D mesh tumorale
  POST /vitro/ic50                 → predici IC50 per linea cellulare
  GET  /alphafold/{gene}           → struttura proteica AlphaFold
  POST /rl/optimize                → lancia RL optimizer (Q-Learning)
  POST /dqn/optimize               → lancia DQN optimizer
  GET  /drugs/compare              → comparazione nanodroni vs farmaci

Autenticazione: Bearer token (mock per ora — in prod usare OAuth2/JWT).
CORS abilitato per dashboard Streamlit.

Avvio: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

# ── Pipeline imports ──────────────────────────────────────────────────────────
from data.gbm_data import GBMDataLoader
from modules.target_analyzer import TargetAnalyzer
from modules.nanodrone_sim import (
    NanodronePKSimulator, FleetConfig, DroneSpec, DeliveryRoute, PKParameters,
    DrugComparison
)
from modules.patient_omics import PatientOmicsLoader
from modules.pk3d_mesh import PK3DMeshSimulator, TumorGeometry, TissuePKParams
from modules.vitro_predictor import VitroPredictor, CELL_LINE_PROFILES
from modules.alphafold_client import AlphaFoldClient
from modules.rl_optimizer import NanodronQAgent, QLearningConfig
from modules.dqn_optimizer import DQNNanodrone, DQNConfig

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARES-Sentinel-GBM API",
    description=(
        "REST API per pipeline computazionale GBM.\n\n"
        "**⚠️ Disclaimer**: Simulazione computazionale a scopo di ricerca. "
        "Non costituisce raccomandazione clinica."
    ),
    version="2.1",
    contact={"name": "ARES GBM Research", "email": "research@ares-gbm.local"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mock auth ─────────────────────────────────────────────────────────────────
API_KEYS = {"ares-dev-2025", "gbm-research-key"}

def verify_token(authorization: Optional[str] = Header(None)):
    """Bearer token check — sostituire con OAuth2 in produzione."""
    if authorization is None:
        return  # Permetti accesso senza token per demo
    if authorization.startswith("Bearer "):
        token = authorization[7:]
        if token in API_KEYS:
            return token
    raise HTTPException(status_code=401, detail="Invalid token")


# ── Modelli Pydantic ──────────────────────────────────────────────────────────

class FleetRequest(BaseModel):
    n_sentinelle:   int   = Field(6,  ge=1, le=20)
    n_decisori:     int   = Field(1,  ge=1, le=10)
    n_attacco:      int   = Field(3,  ge=1, le=20)
    n_lager:        int   = Field(3,  ge=1, le=20)
    delivery_route: str   = Field("FUS", description="FUS|IV_FREE|TRANSFERRIN|CED|LIPOSOME|NP_PLGA|EXOSOME")
    coating:        str   = Field("PEG+Transferrin")
    immune_evasion: float = Field(0.72, ge=0, le=1)
    targeting_affinity: float = Field(0.65, ge=0, le=1)


class PKRequest(BaseModel):
    fleet:      FleetRequest = FleetRequest()
    dose:       float = Field(1.0, gt=0, le=5.0)
    duration_h: float = Field(72.0, ge=1, le=168)
    icp_mmhg:   float = Field(25.0, ge=5, le=50)
    ifp_mmhg:   float = Field(22.0, ge=0, le=40)
    target_gene: str  = Field("EGFR")


class PK3DRequest(BaseModel):
    bee_frac:    float = Field(0.72, ge=0, le=1)
    dose:        float = Field(1.0,  gt=0, le=5.0)
    duration_h:  float = Field(24.0, ge=1, le=72)
    ifp_mmhg:    float = Field(22.0, ge=0, le=40)
    tumor_rx_mm: float = Field(22.0, gt=0)
    tumor_ry_mm: float = Field(18.0, gt=0)
    tumor_rz_mm: float = Field(16.0, gt=0)
    grid_nx:     int   = Field(20,   ge=8, le=40)


class PatientRequest(BaseModel):
    patient_id:  str = Field("PT-001")
    age:         int = Field(58, ge=18, le=100)
    idh_status:  str = Field("wildtype")
    mgmt_status: str = Field("unmethylated")
    seed:        int = Field(42)


class VitroRequest(BaseModel):
    agent:     str = Field("Nano-EGFR")
    cell_line: str = Field("U87")
    n_doses:   int = Field(40, ge=5, le=100)


class RLRequest(BaseModel):
    episodes: int   = Field(300, ge=50, le=1000)
    alpha:    float = Field(0.15, gt=0, le=1.0)


class DQNRequest(BaseModel):
    episodes:  int   = Field(200, ge=50, le=500)
    lr:        float = Field(2e-3, gt=0)
    batch_size: int  = Field(32, ge=8, le=128)


# ── Lazy singletons ──────────────────────────────────────────────────────────
_loader   = None
_mut_df   = None
_expr_df  = None

def _get_data():
    global _loader, _mut_df, _expr_df
    if _loader is None:
        _loader  = GBMDataLoader()
        _mut_df  = _loader.load_mutations()
        _expr_df = _loader.load_expression()
    return _mut_df, _expr_df

def _route_from_str(s: str) -> DeliveryRoute:
    mapping = {r.name: r for r in DeliveryRoute}
    r = mapping.get(s.upper())
    if r is None:
        raise HTTPException(400, f"Route '{s}' non valida. Valori: {list(mapping.keys())}")
    return r

def _build_fleet(req: FleetRequest) -> FleetConfig:
    return FleetConfig(
        n_sentinelle=req.n_sentinelle,
        n_decisori=req.n_decisori,
        n_attacco=req.n_attacco,
        n_lager=req.n_lager,
        delivery_route=_route_from_str(req.delivery_route),
        drone_spec=DroneSpec(
            surface_coating=req.coating,
            immune_evasion=req.immune_evasion,
            targeting_affinity=req.targeting_affinity,
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Stato della pipeline e version info."""
    return {
        "status":   "operational",
        "version":  "2.1",
        "pipeline": "ARES-Sentinel-GBM",
        "disclaimer": "Research use only. Not for clinical decisions.",
        "modules":  ["patient_omics","target_analyzer","nanodrone_sim",
                     "pk3d_mesh","vitro_predictor","alphafold_client",
                     "rl_optimizer","dqn_optimizer"],
    }


@app.get("/targets", tags=["Molecular Analysis"])
async def get_targets(top_n: int = 10, _=Depends(verify_token)):
    """Top target molecolari GBM con composite score."""
    mut_df, expr_df = _get_data()
    analyzer = TargetAnalyzer(mut_df, expr_df)
    score_df = analyzer.score_targets()
    return {
        "top_targets": score_df.head(top_n).to_dict(orient="records"),
        "total_genes": len(score_df),
        "source": "TCGA-GBM n=617 (Brennan et al. Cell 2013)",
    }


@app.get("/targets/synthetic-lethality", tags=["Molecular Analysis"])
async def get_sl(_=Depends(verify_token)):
    """Coppie di letalità sintetica ad alta evidenza per GBM."""
    mut_df, expr_df = _get_data()
    analyzer = TargetAnalyzer(mut_df, expr_df)
    sl_df = analyzer.get_synthetic_lethality_map()
    return sl_df.to_dict(orient="records")


@app.post("/patient/profile", tags=["Patient Omics"])
async def create_patient_profile(req: PatientRequest, _=Depends(verify_token)):
    """Crea profilo paziente da dati WES/scRNA sintetici o caricati."""
    loader  = PatientOmicsLoader(seed=req.seed)
    profile = loader.build_patient_profile(
        patient_id=req.patient_id,
        age=req.age,
        idh_status=req.idh_status,
        mgmt_status=req.mgmt_status,
    )
    return {
        "patient_id":         profile.patient_id,
        "age":                profile.age,
        "idh_status":         profile.idh_status,
        "mgmt_status":        profile.mgmt_status,
        "n_mutations":        len(profile.mutations),
        "top_targets":        profile.top_targets,
        "heterogeneity_score": profile.heterogeneity_score,
        "dominant_cluster":   max(profile.cluster_fractions, key=profile.cluster_fractions.get)
                              if profile.cluster_fractions else "unknown",
        "cluster_fractions":  profile.cluster_fractions,
        "personalized_scores": profile.personalized_score,
        "mutations": [
            {"gene": m.gene, "vaf": m.vaf, "effect": m.effect, "depth": m.depth}
            for m in profile.mutations
        ],
    }


@app.post("/pk/simulate", tags=["PK Simulation"])
async def simulate_pk(req: PKRequest, _=Depends(verify_token)):
    """Simulazione farmacocinetica bicompartimentale plasma → GBM."""
    fleet  = _build_fleet(req.fleet)
    pk_par = PKParameters(bee_base=0.05, tumor_icp=req.icp_mmhg, tumor_ifp=req.ifp_mmhg)
    sim    = NanodronePKSimulator(fleet, pk_par)
    pk_df  = sim.simulate(dose=req.dose, duration_h=req.duration_h,
                           target_gene=req.target_gene)
    summary = sim.pk_summary(pk_df)

    # Campiona 50 punti per non sovraccaricare la risposta
    step = max(1, len(pk_df) // 50)
    sampled = pk_df.iloc[::step][["time_h","c_plasma","c_gbm","effect"]].round(6)

    return {
        "summary":   summary,
        "fleet_info": {
            "total_drones":    fleet.total_drones,
            "delivery_route":  fleet.delivery_route.value,
            "coating":         fleet.drone_spec.surface_coating,
            "effective_bee":   round(fleet.effective_bee(), 4),
        },
        "timeseries": sampled.to_dict(orient="records"),
        "disclaimer": "Computational simulation. Not for clinical use.",
    }


@app.post("/pk/simulate3d", tags=["PK Simulation"])
async def simulate_pk3d(req: PK3DRequest, _=Depends(verify_token)):
    """Simulazione PK 3D su mesh tumorale (equazione di diffusione-convection)."""
    geo = TumorGeometry(
        rx=req.tumor_rx_mm, ry=req.tumor_ry_mm, rz=req.tumor_rz_mm,
        nx=req.grid_nx, ny=max(8, int(req.grid_nx*0.85)),
        nz=max(8, int(req.grid_nx*0.75)),
    )
    pk = TissuePKParams(bee_frac=req.bee_frac, ifp_mmhg=req.ifp_mmhg)
    sim = PK3DMeshSimulator(geo, pk)
    time_df, final_map = sim.simulate(dose=req.dose, duration_h=req.duration_h)
    summary = sim.summary_3d(time_df, final_map)

    step    = max(1, len(time_df) // 60)
    sampled = time_df.iloc[::step][["time_h","c_mean","c_max","c_core","volume_covered_pct"]]

    # Axial slice (z-midplane) per visualizzazione
    axial = sim.axial_slice(final_map, axis=2)

    return {
        "summary":        summary,
        "timeseries":     sampled.to_dict(orient="records"),
        "axial_slice_z":  axial.round(6).tolist(),   # matrice 2D per heatmap
        "disclaimer":     "FDM diffusion model. Mesh approx. Validate in vitro.",
    }


@app.post("/vitro/ic50", tags=["In Vitro"])
async def predict_ic50(req: VitroRequest, _=Depends(verify_token)):
    """Predici IC50 e curva dose-risposta per linea cellulare GBM."""
    if req.cell_line not in CELL_LINE_PROFILES:
        raise HTTPException(400, f"Cell line '{req.cell_line}' non disponibile. "
                                 f"Valori: {list(CELL_LINE_PROFILES.keys())}")
    predictor = VitroPredictor()
    doses = np.logspace(-3, 2, req.n_doses)
    curve = predictor.dose_response_curve(req.agent, req.cell_line, doses)
    ic50_matrix = predictor.compute_ic50_matrix(agents=[req.agent],
                                                 cell_lines=[req.cell_line])
    ic50_row = ic50_matrix.iloc[0] if not ic50_matrix.empty else {}

    return {
        "agent":       req.agent,
        "cell_line":   req.cell_line,
        "subtype":     CELL_LINE_PROFILES[req.cell_line]["subtype"],
        "ic50_uM":     float(ic50_row.get("IC50_uM", 0)),
        "si":          float(ic50_row.get("SI", 0)),
        "emax":        float(ic50_row.get("emax", 0)),
        "resist_prob": float(ic50_row.get("resist_prob", 0)),
        "dose_response": curve[["dose_uM","viability","effect"]].round(4).to_dict(orient="records"),
    }


@app.get("/vitro/cell-lines", tags=["In Vitro"])
async def list_cell_lines(_=Depends(verify_token)):
    """Lista linee cellulari GBM disponibili con profilo molecolare."""
    return [
        {"cell_line": k, "subtype": v["subtype"],
         "egfr": v["egfr_status"], "pten": v["pten_status"],
         "mgmt": v["mgmt_status"], "doubling_time_h": v["doubling_time_h"]}
        for k, v in CELL_LINE_PROFILES.items()
    ]


@app.get("/alphafold/{gene}", tags=["AlphaFold"])
async def get_alphafold(gene: str, _=Depends(verify_token)):
    """Struttura proteica AlphaFold per gene GBM target."""
    client = AlphaFoldClient(timeout_s=6)
    try:
        result = client.get_structure(gene.upper())
        return {
            "gene":             result.gene,
            "uniprot_id":       result.uniprot_id,
            "sequence_len":     result.sequence_len,
            "mean_pLDDT":       result.mean_plddt,
            "high_conf_pct":    result.high_conf_pct,
            "druggability":     "HIGH" if result.high_conf_pct > 60 else
                                "MEDIUM" if result.high_conf_pct > 40 else "LOW",
            "n_druggable_windows": len(result.druggable_regions),
            "druggable_regions": result.druggable_regions[:10],
            "alphafold_url":    result.alphafold_url,
            "source":           result.source,
            "plddt_sample":     result.plddt_scores[:50],   # prime 50 posizioni
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/alphafold/batch/top-targets", tags=["AlphaFold"])
async def alphafold_batch(_=Depends(verify_token)):
    """Query batch AlphaFold per i top-5 target GBM druggabili."""
    client = AlphaFoldClient()
    genes  = ["EGFR", "CDK4", "MDM2", "PIK3CA", "PDGFRA"]
    df     = client.query_targets(genes)
    return df.to_dict(orient="records")


@app.post("/rl/optimize", tags=["RL Optimizer"])
async def run_rl(req: RLRequest, _=Depends(verify_token)):
    """Lancia Q-Learning optimizer per configurazione flotta nanodrone."""
    cfg   = QLearningConfig(episodes=req.episodes, alpha=req.alpha)
    agent = NanodronQAgent(cfg)
    history = agent.train()
    best  = agent.best_config()
    top5  = agent.top_configs(5)

    return {
        "algorithm":  "Q-Learning (tabular)",
        "episodes":   req.episodes,
        "best_config": best,
        "top5":        top5.to_dict(orient="records"),
        "convergence": {
            "final_reward": round(history["reward"].iloc[-10:].mean(), 4),
            "best_reward":  round(history["reward"].max(), 4),
            "best_episode": int(history["reward"].idxmax()) + 1,
        },
    }


@app.post("/dqn/optimize", tags=["RL Optimizer"])
async def run_dqn(req: DQNRequest, _=Depends(verify_token)):
    """Lancia DQN optimizer con Experience Replay e Target Network."""
    cfg   = DQNConfig(episodes=req.episodes, lr=req.lr, batch_size=req.batch_size)
    agent = DQNNanodrone(cfg)
    history = agent.train()
    best  = agent.best_config()
    top5  = agent.top_configs(5)

    return {
        "algorithm":  "DQN (Deep Q-Network, numpy)",
        "n_actions":  2520,
        "architecture": "6 → 64 → 32 → 2520",
        "episodes":   req.episodes,
        "best_config": best,
        "top5":        top5.to_dict(orient="records"),
        "convergence": {
            "final_reward": round(history["reward"].iloc[-10:].mean(), 4),
            "best_reward":  round(history["reward"].max(), 4),
            "mean_loss_last50": round(history["loss"].iloc[-50:].mean(), 6),
        },
    }


@app.get("/drugs/compare", tags=["Drug Comparison"])
async def compare_drugs(_=Depends(verify_token)):
    """Comparazione efficacy score nanodroni vs farmaci convenzionali."""
    comp_df = DrugComparison().compare()
    return {
        "agents":       comp_df.to_dict(orient="records"),
        "best_nano":    comp_df[comp_df["is_nanodrone"]].iloc[0].to_dict(),
        "best_classic": comp_df[~comp_df["is_nanodrone"]].iloc[0].to_dict(),
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
