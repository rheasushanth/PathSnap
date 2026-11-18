"""PathSnap inference API.  OWNER: D   Runs on LAPTOP 1.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

--host 0.0.0.0 is MANDATORY. The uvicorn default of 127.0.0.1 is unreachable
from Laptop 2 and you will lose the demo to it.

STAGE 1 (now):      predict_route returns the raw GPS track as the "route".
                    This exists so the two-laptop path can be proven today,
                    before any model is trained.
STAGE 2 (H+7:00):   swap the body of predict_route for the real pipeline.
                    Nothing else in this file changes.

GROUND TRUTH stays server-side. The response carries aggregate metrics only -
never raw label arrays. That is an explicit rubric requirement.
"""

import os
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.io_utils import list_trips, load_trip

# Point this at the evaluator partition on demo day.
DATA_DIR = os.environ.get("PATHSNAP_DATA", "data/student_test")
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", "artifacts/model.pkl")

app = FastAPI(title="PathSnap API", version="1.0")

# The browser UI on Laptop 2 is a different origin; without this it is blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ARTIFACTS = {"model": None, "features": None, "hmm_cfg": None, "meta": {}}

# Every exported model, loaded lazily and cached: {name: (estimator, features, meta)}
_MODELS = {}
_INDEX = {}
MODELS_INDEX = "artifacts/models.json"


def _load_index():
    """Read artifacts/models.json - the list of models the evaluator can pick."""
    global _INDEX
    if os.path.exists(MODELS_INDEX):
        import json
        with open(MODELS_INDEX) as f:
            _INDEX = json.load(f)
    return _INDEX


def _get_model(name: str):
    """Load a named model, caching it after first use.

    Lazy rather than eager: loading all five at startup costs memory and
    time for models the evaluator may never select.
    """
    if name in _MODELS:
        return _MODELS[name]
    entry = _INDEX.get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown model {name}")
    from src.models import load_final
    bundle = load_final(entry["artifact"])
    meta = {k: v for k, v in bundle.items() if k not in ("model", "features")}
    _MODELS[name] = (bundle["model"], bundle["features"], meta)
    print(f"[{datetime.now():%H:%M:%S}] loaded model: {name}")
    return _MODELS[name]


@app.on_event("startup")
def load_artifacts():
    """Load the model ONCE at startup, never per request."""
    print(f"[{datetime.now():%H:%M:%S}] startup  data={DATA_DIR}")
    if os.path.exists(MODEL_PATH):
        from src.models import load_final
        from src.hmm import load_config
        bundle = load_final(MODEL_PATH)
        _ARTIFACTS["model"] = bundle["model"]
        _ARTIFACTS["features"] = bundle["features"]
        _ARTIFACTS["hmm_cfg"] = load_config()
        _ARTIFACTS["meta"] = {k: v for k, v in bundle.items()
                              if k not in ("model", "features")}
        meta = _ARTIFACTS["meta"]
        print(f"[{datetime.now():%H:%M:%S}] model loaded: {meta.get('model_name')} "
              f"({meta.get('trained_rows', 0):,} rows, "
              f"student-test Top-1 {meta.get('student_test_top1', 0):.4f})")
    else:
        print(f"[{datetime.now():%H:%M:%S}] NO MODEL at {MODEL_PATH} - stage-1 passthrough mode")
    _load_index()
    print(f"[{datetime.now():%H:%M:%S}] selectable models: {list(_INDEX)}")
    print(f"[{datetime.now():%H:%M:%S}] trips available: {len(list_trips(DATA_DIR))}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _ARTIFACTS["model"] is not None,
        "data_dir": DATA_DIR,
        "n_trips": len(list_trips(DATA_DIR)),
    }


@app.get("/trips")
def trips():
    return {"trips": list_trips(DATA_DIR)}


@app.get("/models")
def models():
    """Models the evaluator can choose between, best first.

    Each entry carries its own measured student-test scores, so the UI can
    show what it is choosing between rather than just names.
    """
    idx = _load_index()
    ranked = sorted(idx.values(), key=lambda m: -m["student_test_top1"])
    return {
        "models": [{
            "name": m["model_name"],
            "family": m.get("family"),
            "config": m.get("config"),
            "student_test_top1": m["student_test_top1"],
            "student_test_macro_f1": m.get("student_test_macro_f1"),
            "student_test_log_loss": m.get("student_test_log_loss"),
            "loaded": m["model_name"] in _MODELS,
        } for m in ranked],
        "default": ranked[0]["model_name"] if ranked else None,
    }


@app.get("/comparison")
def comparison():
    """The section 3.1 model-comparison table, in the brief's column order.

    Served from the generated CSV so the UI shows exactly the table that is
    in the report - the evaluator sees the same numbers on screen and on
    paper, with no second transcription to go stale.
    """
    import csv as _csv
    path = "results/table_3_1_model_comparison.csv"
    if not os.path.exists(path):
        return {"columns": [], "rows": [], "note": "table not generated yet"}
    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    return {"columns": reader.fieldnames or [], "rows": rows}


def predict_route(trip_id: str, model_name: str = None, use_hmm: bool = True) -> dict:
    """Run one trip through the full prediction pipeline.

    model_name selects among the exported models; None uses the startup
    default. use_hmm toggles Viterbi decoding, so the evaluator can see the
    classifier-only baseline and the decoded route side by side.

    Falls back to a passthrough echo of the raw GPS when no model artifact is
    present, so the two-laptop path stays demonstrable even without a model.
    """
    trip_dir = os.path.join(DATA_DIR, trip_id)
    if not os.path.isdir(trip_dir):
        raise HTTPException(status_code=404, detail=f"unknown trip {trip_id}")

    t0 = time.time()
    trip = load_trip(trip_dir)
    gps = [[float(lat), float(lon)] for lon, lat, _ in trip["track"]]

    if _ARTIFACTS["model"] is None:
        # Passthrough: echo the GPS so the UI and network path can be verified
        # end to end before a model exists. Metrics are zero because NOTHING
        # was computed - not because a model scored zero.
        return {
            "trip_id": trip_id,
            "stage": "passthrough",
            "model_name": None,
            "gps": gps,
            "predicted_path": gps,
            "predicted_arcs": [],
            "metrics": {
                "top1_accuracy": 0.0,
                "route_continuity": 0.0,
                "n_observations": len(gps),
                "inference_time_s": round(time.time() - t0, 3),
            },
            "classifier_only": {"top1_accuracy": 0.0},
        }

    # --- real pipeline ---------------------------------------------------
    from src.candidates import generate_candidates, _trip_geometry
    from src.features import build_features
    from src.hmm import viterbi
    from src.labels import align_route_to_track, UNLABELLED
    from src.metrics import route_continuity
    from src.viz import arcs_to_latlon_path

    if model_name:
        model, feature_cols, meta = _get_model(model_name)
    else:
        model = _ARTIFACTS["model"]
        feature_cols = _ARTIFACTS["features"]
        meta = _ARTIFACTS["meta"]

    geom = _trip_geometry(trip)
    cands = generate_candidates(trip, meta.get("radius_m", 50.0),
                                meta.get("k", 10), geom=geom)
    feats = build_features(trip, cands, geom=geom)
    feats["proba"] = model.predict_proba(feats[feature_cols])[:, 1]

    n_obs = len(trip["track"])

    # Classifier-only baseline: argmax per observation, no sequential reasoning.
    best = feats.loc[feats.groupby("obs_idx").proba.idxmax()].sort_values("obs_idx")
    base = np.full(n_obs, -1, dtype=np.int64)
    base[best.obs_idx.to_numpy()] = best.arc_id.to_numpy()

    t_hmm = 0.0
    if use_hmm:
        t1 = time.time()
        pred = viterbi(trip, feats[["obs_idx", "arc_id", "proba", "t_along"]],
                       _ARTIFACTS["hmm_cfg"])
        t_hmm = time.time() - t1
    else:
        pred = base

    # --- server-side scoring; ground truth never leaves this function ----
    truth = align_route_to_track(trip)
    scored = truth != UNLABELLED

    def acc(p):
        return float((p[scored] == truth[scored]).mean()) if scored.any() else 0.0

    path = arcs_to_latlon_path(trip, [a for a in pred if a >= 0])

    return {
        "trip_id": trip_id,
        "stage": "model+hmm" if use_hmm else "model",
        "model_name": meta.get("model_name"),
        "model_family": meta.get("family"),
        "hmm_enabled": use_hmm,
        "gps": gps,
        "predicted_path": path,
        "predicted_arcs": [int(a) for a in pred],
        "metrics": {
            "top1_accuracy": round(acc(pred), 4),
            "route_continuity": round(route_continuity(trip, pred), 4),
            "n_observations": n_obs,
            "n_scored": int(scored.sum()),
            "n_candidates": int(len(cands)),
            "hmm_time_s": round(t_hmm, 3),
            "inference_time_s": round(time.time() - t0, 3),
        },
        # Always the independent per-observation argmax, so the UI can show
        # what Viterbi decoding actually bought on this trip.
        "classifier_only": {
            "top1_accuracy": round(acc(base), 4),
            "route_continuity": round(route_continuity(trip, base), 4),
        },
    }


@app.post("/predict/{trip_id}")
def predict(trip_id: str, model: str = None, hmm: bool = True):
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] REQUEST  trip={trip_id} "
          f"model={model or 'default'} hmm={hmm}")
    result = predict_route(trip_id, model_name=model, use_hmm=hmm)
    m = result["metrics"]
    print(f"[{datetime.now():%H:%M:%S}] DONE     trip={trip_id} "
          f"model={result['model_name']} hmm={hmm} "
          f"top1={m['top1_accuracy']:.3f} cont={m['route_continuity']:.3f} "
          f"n={m['n_observations']} t={time.time() - t0:.2f}s")
    return result
