"""Candidate arc generation.  OWNER: Rhea

For each GPS observation, shortlist the road arcs it might be on.

CANDIDATE MISS RATE IS A HARD CEILING. If the true arc is absent from the
shortlist 15% of the time, no model can exceed 85% Top-1 accuracy. Measure and
tune this BEFORE training anything.

Method:
  1. Project the trip once (geo.project_trip), cache it.
  2. cKDTree over arc MIDPOINTS.
  3. Query the nearest MIDPOINT_PREFILTER midpoints per observation, then
     recompute EXACT point-to-segment distance to those arcs and re-rank.
     The midpoint query alone is wrong for long arcs - this trip set has arcs
     up to 2 km long whose midpoint is nowhere near the vehicle - and the
     exact re-rank fixes it.
  4. Keep arcs within radius_m, take the top k by true distance.
  5. If nothing is in range, fall back to the single nearest arc. NEVER emit
     an empty candidate set: Viterbi needs a state at every timestep.
"""

import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.geo import project_trip, arc_endpoints_xy, point_to_segment

DEFAULT_RADIUS_M = 50.0
DEFAULT_K = 10
MIDPOINT_PREFILTER = 40


def _trip_geometry(trip: dict):
    """Projected track, arc endpoints, midpoint KD-tree. Cache per trip."""
    track_xy, node_xy, anchor = project_trip(trip)
    A, B = arc_endpoints_xy(trip, node_xy)
    mid = 0.5 * (A + B)
    return track_xy, node_xy, A, B, cKDTree(mid), anchor


def generate_candidates(trip: dict,
                        radius_m: float = DEFAULT_RADIUS_M,
                        k: int = DEFAULT_K,
                        geom=None) -> pd.DataFrame:
    """One row per (observation, candidate arc).

    Columns: obs_idx, arc_id, dist_m, t_along
    Sorted by obs_idx then dist_m. Pass `geom` to reuse cached geometry.
    """
    if geom is None:
        geom = _trip_geometry(trip)
    track_xy, _, A, B, tree, _ = geom

    n_obs = len(track_xy)
    n_arcs = len(A)
    prefilter = min(MIDPOINT_PREFILTER, n_arcs)

    # (N, prefilter) nearest arcs by midpoint - a generous superset.
    _, cand_idx = tree.query(track_xy, k=prefilter, workers=-1)
    cand_idx = np.atleast_2d(cand_idx)

    # Exact perpendicular distance to each shortlisted arc.
    px = track_xy[:, 0][:, None]
    py = track_xy[:, 1][:, None]
    dist, t = point_to_segment(px, py,
                               A[cand_idx, 0], A[cand_idx, 1],
                               B[cand_idx, 0], B[cand_idx, 1])

    # Re-rank by TRUE distance and keep the best k.
    keep = min(k, prefilter)
    order = np.argsort(dist, axis=1)[:, :keep]
    rows = np.arange(n_obs)[:, None]
    sel_arc = cand_idx[rows, order]
    sel_dist = dist[rows, order]
    sel_t = t[rows, order]

    # Radius filter, but always keep rank 0 so no observation is left empty.
    within = sel_dist <= radius_m
    within[:, 0] = True

    obs_idx = np.repeat(np.arange(n_obs), keep)[within.ravel()]
    return pd.DataFrame({
        "obs_idx": obs_idx.astype(np.int32),
        "arc_id": sel_arc.ravel()[within.ravel()].astype(np.int32),
        "dist_m": sel_dist.ravel()[within.ravel()].astype(np.float32),
        "t_along": sel_t.ravel()[within.ravel()].astype(np.float32),
    })


def candidate_sweep(partition_dir: str,
                    radii=(30, 50, 80, 120),
                    ks=(5, 10, 15, 20),
                    max_trips: int = None) -> pd.DataFrame:
    """Miss-rate sweep -> results/candidate_sweep.csv.

    Pick the SMALLEST radius/k with miss rate under ~2%. Larger k is not free:
    Viterbi cost is O(N*k^2).
    """
    from src.io_utils import list_trips, load_trip
    from src.labels import align_route_to_track, UNLABELLED

    trip_ids = list_trips(partition_dir)
    if max_trips:
        trip_ids = trip_ids[:max_trips]

    cache = []
    for tid in trip_ids:
        trip = load_trip(f"{partition_dir}/{tid}")
        cache.append((trip, _trip_geometry(trip), align_route_to_track(trip)))

    rows = []
    for radius in radii:
        for k in ks:
            t0 = time.time()
            miss, total, n_cand = 0, 0, 0
            for trip, geom, labels in cache:
                cands = generate_candidates(trip, radius, k, geom=geom)
                labelled = np.where(labels != UNLABELLED)[0]
                if not len(labelled):
                    continue
                truth = pd.Series(labels, name="true_arc")
                merged = cands.join(truth, on="obs_idx")
                hit = (merged.arc_id == merged.true_arc).groupby(merged.obs_idx).max()
                hit = hit.reindex(labelled, fill_value=False)
                miss += int((~hit).sum())
                total += len(labelled)
                n_cand += len(cands)
            rows.append({
                "radius_m": radius, "k": k,
                "miss_rate": miss / max(total, 1),
                "mean_candidates_per_obs": n_cand / max(total, 1),
                "n_observations": total,
                "gen_time_s": round(time.time() - t0, 2),
            })
            print(f"  radius={radius:4} k={k:3}  miss={rows[-1]['miss_rate']:.4f}  "
                  f"cands/obs={rows[-1]['mean_candidates_per_obs']:.2f}  "
                  f"{rows[-1]['gen_time_s']:.1f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    partition = sys.argv[1] if len(sys.argv) > 1 else "data/student_test"
    max_trips = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"candidate sweep on {partition}")
    df = candidate_sweep(partition, max_trips=max_trips)
    df.to_csv("results/candidate_sweep.csv", index=False)
    print("\n" + df.to_string(index=False))
    best = df[df.miss_rate < 0.02].nsmallest(1, "mean_candidates_per_obs")
    print("\nsmallest config under 2% miss rate:")
    print(best.to_string(index=False) if len(best) else "  none - widen the sweep")
