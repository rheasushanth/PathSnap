"""Candidate arc generation.  OWNER: B

For each GPS observation, shortlist the road arcs it might be on.
Candidate Miss Rate (how often the true arc is NOT in the shortlist) is a hard
ceiling on final accuracy, so tune radius/k against it before anything else.

Approach:
  1. geo.project_trip once per trip. Cache it.
  2. cKDTree over arc MIDPOINTS.
  3. Query ~40 nearest midpoints per observation, then recompute EXACT
     point_to_segment distance to those arcs and re-rank. The midpoint query
     alone is wrong for long arcs; the exact re-rank fixes it.
  4. Keep arcs within radius_m, take top k by true distance.
  5. If that yields nothing, fall back to the single nearest arc regardless of
     radius. NEVER emit an empty candidate set - Viterbi needs at least one
     state at every timestep.
"""

import numpy as np
import pandas as pd

DEFAULT_RADIUS_M = 50.0
DEFAULT_K = 10
MIDPOINT_PREFILTER = 40


def generate_candidates(trip: dict,
                        radius_m: float = DEFAULT_RADIUS_M,
                        k: int = DEFAULT_K) -> pd.DataFrame:
    """One row per (observation, candidate arc).

    Columns: obs_idx (int), arc_id (int), dist_m (float), t_along (float in [0,1])
    Sorted by obs_idx then dist_m.
    """
    raise NotImplementedError("B")


def candidate_sweep(partition_dir: str,
                    radii=(30, 50, 80, 120),
                    ks=(5, 10, 15, 20)) -> pd.DataFrame:
    """Miss-rate sweep -> results/candidate_sweep.csv, and a table in the report.

    Columns: radius_m, k, miss_rate, mean_candidates_per_obs, gen_time_s

    Pick the SMALLEST radius/k with miss rate under ~2%. Bigger k is not free:
    Viterbi cost is O(N*k^2).
    """
    raise NotImplementedError("B")
