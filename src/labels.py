"""Per-observation label derivation.  OWNER: A

THIS IS THE CRITICAL PATH. Everything downstream depends on it being right.

The problem: .route gives a sequence of arcs (e.g. 123 of them), the .track
gives GPS observations (e.g. 1354 of them). There is no row correspondence.
We must decide, for each GPS observation, which arc of the route it was on.

Algorithm - monotonic alignment by dynamic programming:
  1. Project the trip. Build the route polyline: route position r corresponds
     to arc route[r], occupying cumulative-length interval [s_start, s_end].
  2. D[i, r] = perpendicular distance from observation i to arc route[r].
     Use geo.point_to_segment, vectorised over r.
  3. DP with a no-going-backwards constraint:
         cost[i][r] = D[i][r] + min over r' <= r of (cost[i-1][r'] + SKIP_PENALTY*(r-r'))
     Implement the inner min as a running prefix-minimum so the DP is O(N*R),
     not O(N*R^2).
  4. Backtrack to get the best route position r*[i] per observation.
  5. label[i] = route[r*[i]], or -1 when D[i][r*] > MAX_SNAP_M.

VALIDATE BEFORE MOVING ON (report these numbers in section 2):
  - fraction labelled (expect > 95%)
  - route positions are non-decreasing (assert it - it is guaranteed by the DP,
    so a failure means a bug)
  - median distance to assigned arc (expect ~5-20 m) - plot the histogram
  - plot one trip with GPS points coloured by assigned arc. It must look like a
    clean progression along a road, not confetti.
"""

import numpy as np

SKIP_PENALTY_M = 5.0   # cost per skipped route arc; tune on a few trips
MAX_SNAP_M = 100.0     # beyond this the observation is an outlier -> label -1
UNLABELLED = -1


def align_route_to_track(trip: dict) -> np.ndarray:
    """Per-observation ground-truth arc_id, shape (N,), UNLABELLED where unassignable.

    Returns an all-UNLABELLED array when trip['route'] is None.
    """
    raise NotImplementedError("A: implement the monotonic DP described above")


def align_route_positions(trip: dict) -> np.ndarray:
    """Same as align_route_to_track but returns route POSITION r*, not arc_id.

    Useful for validation (monotonicity) and for debugging plots.
    """
    raise NotImplementedError("A")


def label_quality_report(trip: dict, labels: np.ndarray) -> dict:
    """Validation stats for one trip. Feed these into the EDA notebook."""
    raise NotImplementedError("A")
