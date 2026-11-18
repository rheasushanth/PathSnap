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

_ARTIFACTS = {"model": None, "features": None, "hmm_cfg": None}


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
        print(f"[{datetime.now():%H:%M:%S}] model loaded: {MODEL_PATH}")
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
    """The ONE function that changes between stage 1 and stage 2.

    Stage 2 implementation (D wires this up at H+7:00):
        trip  = load_trip(...)
        cands = generate_candidates(trip, radius_m, k)
        feats = build_features(trip, cands)
        cands['proba'] = model.predict_proba(feats[FEATURE_COLS])[:, 1]
        pred  = viterbi(trip, cands, hmm_cfg)
        base  = classifier_only(cands)
        true  = align_route_to_track(trip)        # server-side only
        metrics from src.metrics.sequence_metrics
    """
    trip_dir = os.path.join(DATA_DIR, trip_id)
    if not os.path.isdir(trip_dir):
        raise HTTPException(status_code=404, detail=f"unknown trip {trip_id}")

    t0 = time.time()
    trip = load_trip(trip_dir)
    gps = [[float(lat), float(lon)] for lon, lat, _ in trip["track"]]

    if _ARTIFACTS["model"] is None:
        # STAGE 1 passthrough: echo the GPS so the UI and network path can be
        # verified end to end before the model exists.
        return {
            "trip_id": trip_id,
            "stage": "passthrough",
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

    raise HTTPException(status_code=501, detail="stage-2 pipeline not wired up yet")


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
