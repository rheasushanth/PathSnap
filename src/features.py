"""Feature engineering.  OWNER: B

One row = one (observation, candidate arc) pair. The model is a BINARY ranker:
"is this candidate the arc the vehicle was actually on?"  Its predicted
probability becomes the HMM emission score.

FEATURE_COLS is a CONTRACT. Freeze it early and tell C immediately - C writes
all the model code against this list.

DO NOT include trip_id, arc_id, obs_idx, raw lon/lat, or anything absolute.
Arc ids are per-trip and coordinates differ by thousands of km between trips;
those features cannot transfer and will mislead the model.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    # geometric
    "dist_m",             # perpendicular distance to the arc
    "dist_rank",          # 0-based rank among this observation's candidates
    "dist_ratio",         # dist_m / (min dist for this obs + eps)
    "dist_gap",           # dist_m - min dist for this obs
    "t_along",            # position along the arc, [0,1]
    "arc_length_m",
    # directional
    "bearing_diff",       # |gps bearing - arc bearing|, folded to [0,180]
    "cos_bearing_diff",   # cos of the above; smoother for linear models
    # kinematic
    "speed_mps",
    "dt_prev",
    "dt_next",
    # topological
    "from_degree",
    "to_degree",
    "connected_to_prev_best",   # shares a node with previous obs' nearest candidate
    # context
    "n_candidates",
]


def build_features(trip: dict, cands: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_COLS to the candidate frame and return it.

    Notes:
      - gps bearing: use observation i-1 -> i+1 (smoothed). Consecutive-point
        bearing is far too noisy at 1 Hz in urban GPS.
      - build the NetworkX graph once per trip for the degree features.
      - all distances in projected metres (see geo.py).
    """
    raise NotImplementedError("B")


def build_dataset(partition_dir: str,
                  radius_m: float = 50.0,
                  k: int = 10,
                  with_labels: bool = True) -> pd.DataFrame:
    """Full matrix across every trip in a partition.

    Columns: trip_id, obs_idx, arc_id, *FEATURE_COLS, label

    label is 1 when arc_id equals the aligned ground-truth arc for that
    observation, else 0. Rows for observations the aligner could not label
    (UNLABELLED) are dropped.

    trip_id is the GroupKFold grouping key - C needs it in the frame.
    Expect roughly 80 trips * ~2000 obs * ~8 candidates = ~1.3M rows.
    Cast features to float32 and save to results/dataset_train.parquet.
    """
    raise NotImplementedError("B")
