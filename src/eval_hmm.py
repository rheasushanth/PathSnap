"""Classifier-only vs classifier+HMM comparison.  OWNER: Rhea

    python -m src.eval_hmm                 # all student-test trips
    python -m src.eval_hmm 00000007        # one trip

Produces results/hmm_comparison.csv, which is report section 5.

Report BOTH directions honestly. If the HMM hurts a trip, keep the row and
explain why - the rubric explicitly asks for cases where decoding did not
help, and large GPS gaps are the usual culprit.
"""

import sys
import time

import numpy as np
import pandas as pd

from src.candidates import _trip_geometry, generate_candidates
from src.features import build_features, FEATURE_COLS
from src.hmm import viterbi, classifier_only, load_config
from src.io_utils import list_trips, load_trip
from src.labels import align_route_to_track, UNLABELLED
from src.metrics import route_continuity
from src.models import load_final


def evaluate_trip(trip, model, feature_cols, cfg, radius_m, k):
    geom = _trip_geometry(trip)
    cands = generate_candidates(trip, radius_m, k, geom=geom)
    feats = build_features(trip, cands, geom=geom)
    feats["proba"] = model.predict_proba(feats[feature_cols])[:, 1]

    truth = align_route_to_track(trip)
    scored = truth != UNLABELLED
    n_obs = len(trip["track"])

    # classifier only
    t0 = time.time()
    base = np.full(n_obs, -1, dtype=np.int64)
    best = feats.loc[feats.groupby("obs_idx").proba.idxmax()]
    base[best.obs_idx.to_numpy()] = best.arc_id.to_numpy()
    t_base = time.time() - t0

    # classifier + HMM
    t0 = time.time()
    decoded = viterbi(trip, feats[["obs_idx", "arc_id", "proba", "t_along"]], cfg)
    t_hmm = time.time() - t0

    def acc(pred):
        return float((pred[scored] == truth[scored]).mean()) if scored.any() else np.nan

    return {
        "trip_id": trip["trip_id"],
        "n_obs": n_obs,
        "n_scored": int(scored.sum()),
        "max_gap_s": float(np.diff(trip["track"][:, 2]).max()),
        "clf_top1": acc(base),
        "hmm_top1": acc(decoded),
        "clf_continuity": route_continuity(trip, base),
        "hmm_continuity": route_continuity(trip, decoded),
        "clf_time_s": round(t_base, 3),
        "hmm_time_s": round(t_hmm, 3),
    }


def main(trip_ids=None, partition="data/student_test"):
    bundle = load_final("artifacts/model.pkl")
    model, feature_cols = bundle["model"], bundle["features"]
    radius_m = bundle.get("radius_m", 50.0)
    k = bundle.get("k", 10)
    cfg = load_config()

    print(f"model={bundle.get('model_name')} radius={radius_m} k={k}")
    print(f"hmm cfg: beta={cfg['beta']} disconnect={cfg['disconnect_penalty']}\n")

    trip_ids = trip_ids or list_trips(partition)
    rows = []
    for tid in trip_ids:
        trip = load_trip(f"{partition}/{tid}")
        r = evaluate_trip(trip, model, feature_cols, cfg, radius_m, k)
        rows.append(r)
        print(f"{tid}  top1 {r['clf_top1']:.4f} -> {r['hmm_top1']:.4f}  "
              f"({r['hmm_top1']-r['clf_top1']:+.4f})   "
              f"cont {r['clf_continuity']:.4f} -> {r['hmm_continuity']:.4f}  "
              f"({r['hmm_continuity']-r['clf_continuity']:+.4f})   "
              f"{r['hmm_time_s']:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df["top1_delta"] = df.hmm_top1 - df.clf_top1
    df["continuity_delta"] = df.hmm_continuity - df.clf_continuity
    df.to_csv("results/hmm_comparison.csv", index=False)

    print("\n=== OVERALL (observation-weighted) ===")
    w = df.n_scored
    for label, c, h in [("Top-1 accuracy", "clf_top1", "hmm_top1"),
                        ("Route continuity", "clf_continuity", "hmm_continuity")]:
        cv = np.average(df[c], weights=w)
        hv = np.average(df[h], weights=w)
        print(f"{label:18}  classifier {cv:.4f}  ->  +HMM {hv:.4f}   ({hv-cv:+.4f})")

    print(f"\ntrips improved on Top-1      : {(df.top1_delta > 0).sum()}/{len(df)}")
    print(f"trips improved on continuity : {(df.continuity_delta > 0).sum()}/{len(df)}")
    worse = df[df.top1_delta < -0.001]
    if len(worse):
        print("\ntrips where HMM HURT accuracy (discuss these in the report):")
        print(worse[["trip_id", "max_gap_s", "clf_top1", "hmm_top1",
                     "top1_delta"]].to_string(index=False))
    print("\nwrote results/hmm_comparison.csv")
    return df


if __name__ == "__main__":
    main(sys.argv[1:] or None)
