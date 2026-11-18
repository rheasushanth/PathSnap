"""Per-observation label derivation.  OWNER: A

THIS IS THE CRITICAL PATH. Everything downstream depends on it being right.

THE PROBLEM
.route gives a sequence of arcs (e.g. 123 of them); .track gives GPS
observations (e.g. 1354 of them). There is no row correspondence. We must
decide, for each GPS observation, which arc of the route the vehicle was on.

THE ALGORITHM - monotonic alignment by dynamic programming
A vehicle traverses its route in order and never jumps backwards. So this is
a monotonic alignment problem (the same shape as DTW or a Viterbi pass):

    cost[i][r] = D[i][r] + min over r' <= r of (cost[i-1][r'] + P*(r - r'))

where D[i][r] is the distance from observation i to route position r, and P
is a penalty per skipped route arc. The constraint r' <= r is what forbids
going backwards.

Done naively the inner min is O(R) and the whole DP is O(N*R^2). We rewrite it
as a running prefix-minimum:

    let m[r] = min over r' <= r of (cost[i-1][r'] - P*r')
    then     min over r' <= r of (cost[i-1][r'] + P*(r-r')) = m[r] + P*r

m[r] is computable in one left-to-right sweep, so the DP is O(N*R).

WHY A PENALTY PER SKIPPED ARC AT ALL
Without it the alignment would happily jump across the whole route to shave a
metre off the distance term. P expresses "staying put or advancing one arc is
normal; skipping twenty arcs between consecutive 1 Hz samples is not".

MEASURED RESULT (all 90 trips, skip_penalty=5 m, max_snap=100 m)
    student_test  99.96% labelled, median snap 4.74 m, all monotonic
    train         98.03% labelled, median snap 4.63 m, all monotonic
    77 of 80 training trips label at 100%.

KNOWN BAD TRIPS - train/00000054 and train/00000033
These label at only 15% and 49%. The cause is NOT the aligner: their track
extent matches their route extent almost exactly, but the whole trajectory is
systematically displaced from the road network.

    trip 54   mean offset (-10.6, +118.2) m   median snap 140.0 m
    trip 33   mean offset (  7.0,  -67.0) m   median snap 122.4 m
    trip  1   mean offset ( -0.2,   +0.2) m   median snap   4.3 m   <- normal

A constant bias of that size is a datum/projection mismatch in the source
data, not GPS noise. Removing the mean offset only brings the residual down
to ~53 m and ~69 m, so these trips are also genuinely noisy.

The MAX_SNAP_M filter already handles them safely: the displaced observations
become UNLABELLED and are dropped before training, so they cannot teach the
model a false road association. Document them in the EDA section as detected
anomalies rather than silently discarding them - a team that finds and
explains these is doing better data analysis than one that does not notice.
"""

import numpy as np

from src.geo import project_trip, point_to_segment

SKIP_PENALTY_M = 5.0   # cost per skipped route arc
MAX_SNAP_M = 100.0     # beyond this the observation is an outlier -> label -1
UNLABELLED = -1


def _route_distance_matrix(trip: dict) -> np.ndarray:
    """D[i, r] = distance in metres from observation i to route position r."""
    track_xy, node_xy, _ = project_trip(trip)
    arcs = trip["arcs"]
    route = trip["route"]

    # Endpoints of each arc in the route, in route order.
    ra = node_xy[arcs[route, 0]]   # (R, 2)
    rb = node_xy[arcs[route, 1]]   # (R, 2)

    # Broadcast every observation against every route position at once:
    # (N,1) against (R,) -> (N,R).
    px = track_xy[:, 0][:, None]
    py = track_xy[:, 1][:, None]
    dist, _ = point_to_segment(px, py,
                               ra[:, 0][None, :], ra[:, 1][None, :],
                               rb[:, 0][None, :], rb[:, 1][None, :])
    return dist


def align_route_positions(trip: dict,
                          skip_penalty_m: float = SKIP_PENALTY_M) -> np.ndarray:
    """Best route POSITION r* for each observation. Shape (N,), always >= 0.

    Returns an all-zero array when the trip has no route.
    Guaranteed non-decreasing - that is the point of the DP.
    """
    n_obs = len(trip["track"])
    route = trip["route"]
    if route is None or len(route) == 0:
        return np.zeros(n_obs, dtype=np.int64)

    D = _route_distance_matrix(trip)
    n, R = D.shape
    P = float(skip_penalty_m)
    pos = np.arange(R, dtype=np.float64)

    # Starting late on the route is itself penalised, so a trip that begins at
    # route position 0 is not charged for it.
    cost = D[0] + P * pos
    back = np.empty((n, R), dtype=np.int32)
    back[0] = np.arange(R)

    for i in range(1, n):
        shifted = cost - P * pos                 # cost[i-1][r'] - P*r'
        m = np.minimum.accumulate(shifted)       # running prefix minimum
        # argmin of the prefix: index where the running min last changed.
        is_new_min = shifted <= m
        src = np.maximum.accumulate(np.where(is_new_min, np.arange(R), -1))
        cost = D[i] + m + P * pos
        back[i] = src

    # Backtrack from the cheapest terminal state.
    r = int(np.argmin(cost))
    out = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[i] = r
        r = int(back[i][r])
    return out


def align_route_to_track(trip: dict,
                         skip_penalty_m: float = SKIP_PENALTY_M,
                         max_snap_m: float = MAX_SNAP_M) -> np.ndarray:
    """Per-observation ground-truth arc_id, shape (N,).

    UNLABELLED (-1) where the trip has no route, or where the assigned arc is
    further than max_snap_m away - those observations are GPS outliers and
    training on them would teach the model noise.
    """
    n_obs = len(trip["track"])
    route = trip["route"]
    if route is None or len(route) == 0:
        return np.full(n_obs, UNLABELLED, dtype=np.int64)

    positions = align_route_positions(trip, skip_penalty_m)
    labels = route[positions].astype(np.int64)

    D = _route_distance_matrix(trip)
    snap_dist = D[np.arange(n_obs), positions]
    labels[snap_dist > max_snap_m] = UNLABELLED
    return labels


def label_quality_report(trip: dict,
                         labels: np.ndarray = None,
                         skip_penalty_m: float = SKIP_PENALTY_M) -> dict:
    """Validation stats for one trip. Feed these into the EDA notebook.

    Healthy trip: labelled_frac > 0.95, median_snap_m in roughly 5-20 m,
    monotonic True, and route_coverage close to 1 (most route arcs actually
    got used - a low value means the alignment collapsed onto a few arcs).
    """
    route = trip["route"]
    n_obs = len(trip["track"])
    if route is None or len(route) == 0:
        return {"trip_id": trip["trip_id"], "n_obs": n_obs, "has_route": False,
                "labelled_frac": 0.0, "monotonic": True}

    positions = align_route_positions(trip, skip_penalty_m)
    if labels is None:
        labels = align_route_to_track(trip, skip_penalty_m)

    D = _route_distance_matrix(trip)
    snap = D[np.arange(n_obs), positions]
    labelled = labels != UNLABELLED

    return {
        "trip_id": trip["trip_id"],
        "n_obs": n_obs,
        "has_route": True,
        "n_route_arcs": len(route),
        "labelled_frac": float(labelled.mean()),
        "n_unlabelled": int((~labelled).sum()),
        "median_snap_m": float(np.median(snap)),
        "p95_snap_m": float(np.percentile(snap, 95)),
        "max_snap_m": float(snap.max()),
        "monotonic": bool(np.all(np.diff(positions) >= 0)),
        "route_coverage": float(len(np.unique(positions)) / len(route)),
        "n_distinct_arcs": int(len(np.unique(labels[labelled]))),
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    from src.io_utils import list_trips, load_trip

    partition = sys.argv[1] if len(sys.argv) > 1 else "data/student_test"
    rows = [label_quality_report(load_trip(f"{partition}/{t}"))
            for t in list_trips(partition)]
    df = pd.DataFrame(rows)
    out = f"results/label_quality_{partition.replace('/', '_')}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    print(f"mean labelled fraction: {df.labelled_frac.mean():.4f}")
    print(f"median snap distance:   {df.median_snap_m.median():.2f} m")
    print(f"all monotonic:          {df.monotonic.all()}")
