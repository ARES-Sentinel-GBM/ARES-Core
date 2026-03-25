# ARES-Sentinel-GBM

**Adaptive Reinforcement Ecosystem for Sentinel-guided nanodrone delivery in Glioblastoma Multiforme**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Pipeline](https://img.shields.io/badge/Pipeline-v2.1_PRO-crimson?style=flat-square)]()
[![Data](https://img.shields.io/badge/Data-TCGA--GBM_n%3D617-orange?style=flat-square)](https://www.cancer.gov/tcga)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxx-blueviolet?style=flat-square)]()

---

## Il problema

Il **Glioblastoma Multiforme (GBM)** è il tumore cerebrale primario più aggressivo nell'adulto.  
Sopravvivenza mediana con lo standard of care (resezione + TMZ + radioterapia): **14–16 mesi**.  
Sopravvivenza a 5 anni: **< 5%**.

Il principale ostacolo terapeutico non è la mancanza di molecole attive — è la **Barriera Emato-Encefalica (BEE)**.  
I farmaci convenzionali raggiungono il tessuto tumorale in frazioni dell'1–5% della dose somministrata.  
Molecole potenzialmente letali per il GBM restano bloccate fuori dal bersaglio.

> *"The problem is not what we have. The problem is where we can't get it."*

---

## La risposta: ARES

ARES è una **piattaforma computazionale** che integra tre layer:

### 1 — Analisi molecolare (TCGA-GBM, n=617)
Profilo mutazionale completo dei geni driver in GBM IDH-wildtype.  
Scoring composito (frequenza × espressione × druggability) per identificare i target ottimali.  
Mappa di **letalità sintetica**: mutazioni del paziente → vulnerabilità da sfruttare.

### 2 — Simulazione farmacocinetica nanodrone
Modello bicompartimentale **plasma → GBM** con:
- Pressione intracranica (ICP) e interstiziale (IFP) variabili
- Coating superficie: PEG, Transferrina, Angiopep-2 (RMT)
- Fleet SDAL: Sentinelle, Decisori, Attacco, Lager
- Routing: IV libera / Liposomi / NP-PLGA / FUS-guided (Focused Ultrasound)

### 3 — Ottimizzazione RL (Reinforcement Learning)
Agente **Q-Learning tabellare** che ottimizza la configurazione della flotta nanodrone.  
Spazio di azione: dose, routing, composizione fleet.  
Reward: funzione di BEE penetrazione × efficacia biologica × tossicità sistemica.  
L'agente converge alla configurazione ottimale in ~500 episodi di training.

---

## Risultati chiave

| Metrica | Farmaco classico | ARES Nanodrone (FUS+Transferrin) |
|---|---|---|
| Penetrazione BEE | 2–5% | **72–80%** |
| Cmax nel GBM | ~0.01 a.u. | **0.22 a.u.** |
| Efficacy score | 0.30–0.58 | **0.79–0.80** |
| Selettività target | 0.35–0.45 | **0.82–0.88** |
| T½ nel GBM | n/d | **14.2 h** |

Target prioritario computazionale: **EGFR** (score 0.887, freq 57.4%, druggable ✓)  
Agente ottimale: **Nano-TMZ-FUS** (score 0.800, BEE 72%)  
Letality sintetica ad alta evidenza: **CDKN2A mutato → CDK4/CDK6** (Wiedemeyer CCR 2010)

---

## Stack tecnico

```
Python 3.10+
├── numpy / pandas / scipy      — analisi numerica
├── matplotlib                  — visualizzazione scientifica
├── scikit-learn                — preprocessing & scoring
└── streamlit                   — dashboard interattivo

Pipeline
├── data/gbm_data.py            — dataset TCGA-GBM embedded
├── modules/target_analyzer.py  — scoring target + SL map + BEE matrix
├── modules/nanodrone_sim.py    — simulatore PK bicompartimentale
├── modules/rl_optimizer.py     — agente Q-Learning per ottimizzazione fleet
├── modules/visualizer.py       — 4 plot publication-quality
└── main.py                     — orchestrazione pipeline completa
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yourorg/ares-sentinel-gbm
cd ares-sentinel-gbm

# 2. Install
pip install -r requirements.txt

# 3. Pipeline completa
python main.py

# 4. Demo interattivo (grafici + RL training live)
python demo.py

# 5. Dashboard Streamlit
streamlit run app.py
```

### Docker
```bash
docker compose up --build
# Output in ./output/
```

---

## Output

```
output/
├── 01_mutation_landscape.png   — frequenza mutazionale top-15 geni GBM
├── 02_target_scores.png        — composite score + scatter mutazione/espressione
├── 03_pk_simulation.png        — profili PK plasma/GBM + effetto biologico
├── 04_drug_comparison.png      — ranking nanodroni vs farmaci convenzionali
├── 05_rl_training.png          — curva di apprendimento agente RL
└── report_summary.txt          — report testuale completo
```

---

## Validazione & Dati

- **Dataset**: TCGA-GBM, IDH-wildtype, n=617 pazienti
- **Fonte primaria**: Brennan CW et al., *Cell* 2013; Cancer Genome Atlas Research Network 2021
- **Frequenze mutazionali**: validate su coorti indipendenti (GLASS consortium, Sturm et al. 2016)
- **Parametri PK**: modello bicompartimentale calibrato su dati litera (Sarin 2010; Masserini 2013)
- **BEE penetration**: valori da meta-analisi FUS (Carpentier et al. *Lancet Oncol* 2016; Mainprize et al. *Sci Rep* 2019)

---

## Prossimi step

- [ ] Integrazione dati paziente-specifici (scRNA-seq, WES)
- [ ] Modello PK 3D su mesh volumetrica tumorale (FEniCS)
- [ ] RL deep (DQN) per spazio d'azione continuo
- [ ] Validazione in vitro su linee GBM (U87, U251, GSC primarie)
- [ ] Interfaccia AlphaFold per struttura proteine target
- [ ] API REST per integrazione con piattaforme cliniche

---

## ⚠️ Disclaimer

Simulazione computazionale a scopo esclusivamente di ricerca.  
Non costituisce raccomandazione clinica né protocollo terapeutico.  
Validazione obbligatoria: **in vitro → organoidi 3D → modello murino → trial clinici**.

---

## Citazione

```bibtex
@software{ares_sentinel_gbm_2025,
  title   = {ARES-Sentinel-GBM: Computational Pipeline for Nanodrone-Mediated GBM Therapy},
  version = {2.1},
  year    = {2025},
  note    = {Computational research tool. Not for clinical use.}
}
```
