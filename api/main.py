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


def predict_route(trip_id: str) -> dict:
    """Run one trip through the full prediction pipeline.

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
    from src.labels import align_route_to_track, UNLABELLED
    from src.metrics import route_continuity
    from src.viz import arcs_to_latlon_path

    model = _ARTIFACTS["model"]
    feature_cols = _ARTIFACTS["features"]
    meta = _ARTIFACTS["meta"]

    geom = _trip_geometry(trip)
    cands = generate_candidates(trip, meta.get("radius_m", 50.0),
                                meta.get("k", 10), geom=geom)
    feats = build_features(trip, cands, geom=geom)
    feats["proba"] = model.predict_proba(feats[feature_cols])[:, 1]

    # Argmax candidate per observation. Observations with no candidate keep -1.
    best = feats.loc[feats.groupby("obs_idx").proba.idxmax()].sort_values("obs_idx")
    pred = np.full(len(trip["track"]), -1, dtype=np.int64)
    pred[best.obs_idx.to_numpy()] = best.arc_id.to_numpy()

    # --- server-side scoring; ground truth never leaves this function ----
    truth = align_route_to_track(trip)
    scored = truth != UNLABELLED
    top1 = float((pred[scored] == truth[scored]).mean()) if scored.any() else 0.0
    continuity = route_continuity(trip, pred)

    path = arcs_to_latlon_path(trip, [a for a in pred if a >= 0])

    return {
        "trip_id": trip_id,
        "stage": "model",
        "model_name": meta.get("model_name"),
        "gps": gps,
        "predicted_path": path,
        "predicted_arcs": [int(a) for a in pred],
        "metrics": {
            "top1_accuracy": round(top1, 4),
            "route_continuity": round(continuity, 4),
            "n_observations": len(gps),
            "n_scored": int(scored.sum()),
            "n_candidates": int(len(cands)),
            "inference_time_s": round(time.time() - t0, 3),
        },
        # Identical here because no decoder runs yet. Once Viterbi lands,
        # top1_accuracy becomes the decoded route and this stays the
        # independent per-observation argmax, making the gain visible.
        "classifier_only": {"top1_accuracy": round(top1, 4)},
    }


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] REQUEST  trip={trip_id}")
    result = predict_route(trip_id)
    m = result["metrics"]
    print(f"[{datetime.now():%H:%M:%S}] DONE     trip={trip_id} "
          f"top1={m['top1_accuracy']:.3f} cont={m['route_continuity']:.3f} "
          f"n={m['n_observations']} t={time.time() - t0:.2f}s")
    return result
