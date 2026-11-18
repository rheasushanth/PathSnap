"""Feature engineering.  OWNER: Rhea

One row = one (observation, candidate arc) pair. The model is a BINARY ranker:
"is this candidate the arc the vehicle was actually on?"  Its predicted
probability becomes the HMM emission score.

FEATURE_COLS is a CONTRACT - the API loads a model trained on exactly this
list, in this order.

Nothing absolute is allowed in here: no trip_id, arc_id, obs_idx or raw
lon/lat. Arc ids are per-trip and the trips are thousands of km apart, so
absolute features cannot transfer and would leak trip identity.
"""

import time

import numpy as np
import pandas as pd

from src.candidates import generate_candidates, _trip_geometry
from src.geo import bearing_deg, angle_diff_deg
from src.labels import align_route_to_track, UNLABELLED

GEOMETRIC = ["dist_m", "dist_rank", "dist_ratio", "dist_gap", "t_along", "arc_length_m"]
DIRECTIONAL = ["bearing_diff", "cos_bearing_diff"]
KINEMATIC = ["speed_mps", "dt_prev", "dt_next"]
TOPOLOGICAL = ["from_degree", "to_degree", "connected_to_prev_best"]
CONTEXT = ["n_candidates"]

FEATURE_COLS = GEOMETRIC + DIRECTIONAL + KINEMATIC + TOPOLOGICAL + CONTEXT

FEATURE_GROUPS = {
    "geometric": GEOMETRIC,
    "directional": DIRECTIONAL,
    "kinematic": KINEMATIC,
    "topological": TOPOLOGICAL,
    "context": CONTEXT,
}


def build_features(trip: dict, cands: pd.DataFrame, geom=None) -> pd.DataFrame:
    """Add FEATURE_COLS to the candidate frame and return it."""
    if geom is None:
        geom = _trip_geometry(trip)
    track_xy, _, A, B, _, _ = geom

    arcs = trip["arcs"]
    ts = trip["track"][:, 2]
    n_obs = len(track_xy)
    obs = cands.obs_idx.to_numpy()
    arc = cands.arc_id.to_numpy()

    df = cands.copy()

    # --- geometric -------------------------------------------------------
    grp = df.groupby("obs_idx")["dist_m"]
    df["dist_rank"] = grp.rank(method="first").to_numpy() - 1
    min_d = grp.transform("min").to_numpy()
    df["dist_ratio"] = df.dist_m.to_numpy() / (min_d + 1e-6)
    df["dist_gap"] = df.dist_m.to_numpy() - min_d
    df["arc_length_m"] = np.hypot(B[arc, 0] - A[arc, 0], B[arc, 1] - A[arc, 1])

    # --- directional -----------------------------------------------------
    # Smoothed GPS heading: i-1 -> i+1. A consecutive-point bearing at 1 Hz in
    # urban GPS is dominated by noise, since the vehicle may move under a
    # metre between samples while the error is several metres.
    prev_i = np.clip(np.arange(n_obs) - 1, 0, n_obs - 1)
    next_i = np.clip(np.arange(n_obs) + 1, 0, n_obs - 1)
    gps_bear = bearing_deg(track_xy[prev_i, 0], track_xy[prev_i, 1],
                           track_xy[next_i, 0], track_xy[next_i, 1])
    arc_bear = bearing_deg(A[arc, 0], A[arc, 1], B[arc, 0], B[arc, 1])
    bdiff = angle_diff_deg(gps_bear[obs], arc_bear)
    df["bearing_diff"] = bdiff
    df["cos_bearing_diff"] = np.cos(np.radians(bdiff))

    # --- kinematic -------------------------------------------------------
    step = np.hypot(np.diff(track_xy[:, 0], prepend=track_xy[0, 0]),
                    np.diff(track_xy[:, 1], prepend=track_xy[0, 1]))
    dt_prev = np.diff(ts, prepend=ts[0] - 1.0)
    dt_next = np.diff(ts, append=ts[-1] + 1.0)
    speed = step / np.maximum(dt_prev, 1e-6)
    df["speed_mps"] = np.clip(speed[obs], 0, 60)
    df["dt_prev"] = np.clip(dt_prev[obs], 0, 300)
    df["dt_next"] = np.clip(dt_next[obs], 0, 300)

    # --- topological -----------------------------------------------------
    degree = np.bincount(arcs.ravel(), minlength=len(trip["nodes"]))
    df["from_degree"] = degree[arcs[arc, 0]]
    df["to_degree"] = degree[arcs[arc, 1]]

    # Cheap sequential hint: does this candidate touch the previous
    # observation's NEAREST arc? Uses geometry only, never the label.
    nearest = (df.loc[df.dist_rank == 0, ["obs_idx", "arc_id"]]
                 .set_index("obs_idx").arc_id.reindex(range(n_obs)).ffill().bfill()
                 .to_numpy().astype(np.int64))
    prev_arc = nearest[prev_i][obs]
    pf, pt = arcs[prev_arc, 0], arcs[prev_arc, 1]
    cf, ct = arcs[arc, 0], arcs[arc, 1]
    df["connected_to_prev_best"] = (
        (arc == prev_arc) | (cf == pf) | (cf == pt) | (ct == pf) | (ct == pt)
    ).astype(np.int8)

    # --- context ---------------------------------------------------------
    df["n_candidates"] = df.groupby("obs_idx")["arc_id"].transform("size").to_numpy()

    for c in FEATURE_COLS:
        df[c] = df[c].astype(np.float32)
    return df


def build_trip_dataset(trip: dict, radius_m=50.0, k=10,
                       with_labels=True, geom=None) -> pd.DataFrame:
    """Feature matrix for a single trip, with a `label` column when available."""
    if geom is None:
        geom = _trip_geometry(trip)
    cands = generate_candidates(trip, radius_m, k, geom=geom)
    df = build_features(trip, cands, geom=geom)
    df.insert(0, "trip_id", trip["trip_id"])

    if with_labels:
        labels = align_route_to_track(trip)
        true_arc = labels[df.obs_idx.to_numpy()]
        df["label"] = (df.arc_id.to_numpy() == true_arc).astype(np.int8)
        df = df[true_arc != UNLABELLED]          # drop unlabelled observations
    return df.reset_index(drop=True)


def build_dataset(partition_dir: str, radius_m=50.0, k=10,
                  with_labels=True, max_trips=None, out_path=None) -> pd.DataFrame:
    """Full matrix across every trip in a partition.

    Columns: trip_id, obs_idx, arc_id, *FEATURE_COLS, label
    trip_id is the GroupKFold grouping key.
    """
    from src.io_utils import list_trips, load_trip

    trip_ids = list_trips(partition_dir)
    if max_trips:
        trip_ids = trip_ids[:max_trips]

    frames, t0 = [], time.time()
    for n, tid in enumerate(trip_ids, 1):
        trip = load_trip(f"{partition_dir}/{tid}")
        frames.append(build_trip_dataset(trip, radius_m, k, with_labels))
        if n % 10 == 0 or n == len(trip_ids):
            print(f"  {n}/{len(trip_ids)} trips  {time.time()-t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    if out_path:
        save_dataset(df, out_path)
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Pickle by default. Parquet needs pyarrow, which on this machine needs
    NumPy >= 2.0 while the rest of the stack is pinned to 1.26 - not a fight
    worth having mid-hackathon. Pickle is fast, lossless and dependency-free.
    """
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)
    print(f"wrote {path} ({len(df):,} rows)")


def load_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_pickle(path)


if __name__ == "__main__":
    import sys
    partition = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/dataset_train.pkl"
    print(f"building dataset from {partition}")
    df = build_dataset(partition, out_path=out)
    print(f"\nrows {len(df):,}  positives {df.label.mean():.4f}  trips {df.trip_id.nunique()}")
    print(df[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(2).to_string())
