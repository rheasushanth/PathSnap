"""Raw dataset loading.  OWNER: A

File formats (all whitespace-separated, NO headers):
    .track  3 cols: longitude, latitude, timestamp_seconds
    .nodes  2 cols: longitude, latitude      -> row index IS the node_id
    .arcs   2 cols: from_node_id, to_node_id -> row index IS the arc_id
    .route  1 col : chronological sequence of ground-truth arc_ids

Note .route is a sequence of ARCS, not one label per GPS observation.
See src/labels.py for turning it into per-observation labels.
"""

import os
import numpy as np

TRACK, NODES, ARCS, ROUTE = ".track", ".nodes", ".arcs", ".route"


def list_trips(partition_dir: str) -> list[str]:
    """Sorted trip ids (directory names) inside a partition such as data/train."""
    if not os.path.isdir(partition_dir):
        raise FileNotFoundError(f"partition not found: {partition_dir}")
    return sorted(
        d for d in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, d))
    )


def load_trip(trip_dir: str) -> dict:
    """Load one trip directory.

    Returns dict with keys:
        trip_id  str
        track    (N, 3) float   lon, lat, timestamp_seconds
        nodes    (M, 2) float   lon, lat          (row index = node_id)
        arcs     (K, 2) int     from_id, to_id    (row index = arc_id)
        route    (R,)   int  or None              chronological arc_ids

    route is None when the file is absent - that is expected for the
    evaluator partition, so callers must handle it.
    """
    trip_dir = os.path.normpath(trip_dir)
    trip_id = os.path.basename(trip_dir)

    def _path(ext):
        return os.path.join(trip_dir, trip_id + ext)

    track = np.loadtxt(_path(TRACK), dtype=np.float64, ndmin=2)
    nodes = np.loadtxt(_path(NODES), dtype=np.float64, ndmin=2)
    arcs = np.loadtxt(_path(ARCS), dtype=np.int64, ndmin=2)

    route_path = _path(ROUTE)
    route = None
    if os.path.exists(route_path):
        route = np.atleast_1d(np.loadtxt(route_path, dtype=np.int64))

    trip = {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }
    validate_trip(trip)
    return trip


def validate_trip(trip: dict) -> list[str]:
    """Return a list of human-readable problems. Empty list == clean trip.

    Does not raise: some trips are expected to be imperfect and the EDA
    section of the report needs a count of exactly which ones and why.
    """
    problems = []
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]

    if track.shape[1] != 3:
        problems.append(f"track has {track.shape[1]} cols, expected 3")
    if nodes.shape[1] != 2:
        problems.append(f"nodes has {nodes.shape[1]} cols, expected 2")
    if arcs.shape[1] != 2:
        problems.append(f"arcs has {arcs.shape[1]} cols, expected 2")

    if not np.isfinite(track).all():
        problems.append("track contains NaN/inf")

    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    if np.any(np.abs(lon) > 180) or np.any(np.abs(lat) > 90):
        problems.append("coordinates out of range - lon/lat may be swapped")

    dt = np.diff(ts)
    if np.any(dt < 0):
        problems.append(f"{int((dt < 0).sum())} non-monotonic timestamps")
    if np.any(dt == 0):
        problems.append(f"{int((dt == 0).sum())} duplicate timestamps")

    n_nodes = len(nodes)
    if arcs.size and (arcs.min() < 0 or arcs.max() >= n_nodes):
        problems.append("arc references a node_id outside the nodes file")

    if route is not None and len(route):
        if route.min() < 0 or route.max() >= len(arcs):
            problems.append("route references an arc_id outside the arcs file")

    return problems


def trip_summary(trip: dict) -> dict:
    """One row of the dataset-inventory table for report section 2."""
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    dt = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    return {
        "trip_id": trip["trip_id"],
        "n_obs": len(track),
        "n_nodes": len(nodes),
        "n_arcs": len(arcs),
        "n_route_arcs": 0 if route is None else len(route),
        "has_route": route is not None,
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "dt_median": float(np.median(dt)),
        "dt_max": float(dt.max()),
        "n_gaps_gt_5s": int((dt > 5).sum()),
        "n_problems": len(validate_trip(trip)),
    }
