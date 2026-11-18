# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning · PES University, RR Campus

Reconstructs realistic vehicle trajectories from noisy GPS by matching observations
to a road network, using a learned candidate ranker as the emission model inside an
HMM decoded with Viterbi.

## Team

| Member | Owns |
|---|---|
| A | Data loading, label alignment, EDA, Folium maps |
| B | Candidate generation, feature engineering, HMM + Viterbi |
| C | Model experimentation, Bayesian optimisation, comparison plots |
| D | FastAPI backend, evaluator UI, deployment, report assembly |

*(replace with real names before submitting)*

## Approach

Each trip ships with its **own** road network, so arc IDs are only meaningful within
a trip and a global arc classifier is impossible. Instead:

1. **Label derivation** — `.route` is a sequence of arcs, not per-observation labels.
   A monotonic DP aligns GPS observations to the route's arc sequence.
2. **GPS cleaning** — multi-factor anomaly scoring with iterative one-at-a-time
   outlier removal, stall compression, and constant-velocity Kalman smoothing.
3. **Candidate generation** — KD-tree prefilter over arc midpoints, exact
   point-to-segment re-rank, top *k* within a radius.
4. **Binary candidate ranker** — one row per (observation, candidate arc); the model
   predicts whether that candidate is the true arc. Probabilities are **calibrated**,
   since they serve as HMM emissions.
5. **HMM + Viterbi** — calibrated classifier probability as the emission; transition
   scores compare network shortest-path distance against GPS step distance, with a
   speed- and gap-adaptive scale.
6. **Confidence scoring** — per-trip High/Medium/Low label with a nearest-arc
   fallback, so a bad decode never degrades the output below the trivial baseline.

Distances use a per-trip local equirectangular projection to metres (the dataset spans
Germany to Russia, so no single UTM zone applies).

The core framework is Newson & Krumm (2009) HMM map matching with a learned emission
model. The cleaning pipeline and confidence-scoring design are adapted from Zepto's
[Route IQ](https://blog.zepto.com/route-iq-how-we-reconstruct-delivery-paths-from-noisy-gps-at-scale-6898d3260ed2)
engineering blog (Aug 2026). Every addition beyond the base framework is reported with
a measured on/off ablation — see `results/ablation.csv`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Extract the dataset so it looks like:

```
data/
├── train/           80 trips
├── student_test/    10 trips
└── evaluator_test/  10 trips (demo day only)
```

## Running the system

**Laptop 1 — API**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required; the default `127.0.0.1` is unreachable from Laptop 2.
Find the machine's address with `ipconfig` and allow inbound TCP 8000 through
Windows Firewall.

**Laptop 2 — UI**

```bash
streamlit run ui/app.py --server.port 8501
```

Enter Laptop 1's address (e.g. `http://192.168.1.37:8000`) in the API URL box.

Point the API at the evaluator partition on demo day:

```bash
set PATHSNAP_DATA=data/evaluator_test
```

## Reproducing the results

```bash
python -m src.features        # build training matrix  -> results/dataset_train.parquet
python -m src.models          # train and compare 5 models -> results/model_comparison.csv
python -m src.tune            # Bayesian optimisation  -> results/optuna_trials.csv
python -m src.hmm             # decoder tuning         -> artifacts/hmm_config.json
```

Notebooks in `notebooks/` cover EDA (`01`), model comparison (`02`) and HMM
evaluation (`03`).

## Layout

```
src/       io_utils, geo, labels, candidates, features, hmm, models, tune, metrics, viz
api/       FastAPI backend (Laptop 1)
ui/        Streamlit evaluator interface (Laptop 2)
notebooks/ EDA and experiment notebooks
artifacts/ trained model, scaler, HMM config
results/   comparison tables, optuna trials, plots
maps/      exported Folium HTML
```

`src/fake.py` generates correctly-shaped synthetic data so any member can build
against an interface before its owner has finished it.

## Data and integrity notes

- The evaluator partition is used for final evaluation only — never for training,
  tuning, feature selection or threshold choice.
- Ground-truth labels are read server-side and never returned to the UI beyond
  aggregate metrics.
- No predictions are hardcoded; every result comes from the live pipeline.

See [PLAN.md](PLAN.md) for the full execution plan.
