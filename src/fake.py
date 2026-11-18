"""Synthetic data so nobody is ever blocked.  OWNER: shared

Rule from PLAN.md section "During the day": if you are waiting on someone
else's module, generate fake data with the right SHAPE and write your code
against it. Swap in the real thing at the sync point.

C can build the entire training and tuning pipeline against fake_dataset()
before B has written a single feature.
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS


def fake_dataset(n_trips: int = 20,
                 obs_per_trip: int = 200,
                 k: int = 8,
                 seed: int = 0) -> pd.DataFrame:
    """A DataFrame shaped exactly like features.build_dataset() output.

    Columns: trip_id, obs_idx, arc_id, *FEATURE_COLS, label
    Exactly one positive per observation (so Top-1 accuracy is meaningful),
    and dist_m is made weakly predictive so models score above chance.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        trip_id = f"fake{t:05d}"
        for i in range(obs_per_trip):
            true_slot = rng.integers(0, k)
            for c in range(k):
                is_true = int(c == true_slot)
                # true candidate tends to be closer, with heavy overlap
                dist = abs(rng.normal(8 if is_true else 28, 12))
                rows.append({
                    "trip_id": trip_id,
                    "obs_idx": i,
                    "arc_id": int(rng.integers(0, 4000)),
                    "dist_m": dist,
                    "dist_rank": c,
                    "dist_ratio": dist / 8.0,
                    "dist_gap": dist - 8.0,
                    "t_along": rng.random(),
                    "arc_length_m": abs(rng.normal(60, 30)),
                    "bearing_diff": abs(rng.normal(10 if is_true else 70, 40)) % 180,
                    "cos_bearing_diff": rng.random(),
                    "speed_mps": abs(rng.normal(12, 5)),
                    "dt_prev": 1.0,
                    "dt_next": 1.0,
                    "from_degree": rng.integers(1, 5),
                    "to_degree": rng.integers(1, 5),
                    "connected_to_prev_best": int(rng.random() < (0.8 if is_true else 0.2)),
                    "n_candidates": k,
                    "label": is_true,
                })
    df = pd.DataFrame(rows)
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise AssertionError(f"fake_dataset out of sync with FEATURE_COLS: {missing}")
    return df


def fake_trip(n_obs: int = 300, n_nodes: int = 200, seed: int = 0) -> dict:
    """A trip dict shaped like io_utils.load_trip() output, for viz/HMM testing."""
    rng = np.random.default_rng(seed)
    lon0, lat0 = 77.12, 43.39
    nodes = np.column_stack([lon0 + rng.normal(0, 0.01, n_nodes),
                             lat0 + rng.normal(0, 0.01, n_nodes)])
    arcs = np.column_stack([rng.integers(0, n_nodes, n_nodes * 2),
                            rng.integers(0, n_nodes, n_nodes * 2)])
    t = np.arange(n_obs, dtype=float)
    track = np.column_stack([lon0 + t * 1e-5 + rng.normal(0, 2e-5, n_obs),
                             lat0 + t * 1e-5 + rng.normal(0, 2e-5, n_obs),
                             t + 1.0])
    return {"trip_id": "fake00000", "track": track, "nodes": nodes,
            "arcs": arcs, "route": rng.integers(0, len(arcs), 40)}
