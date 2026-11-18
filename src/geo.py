"""Geographic helpers.  OWNER: A

WHY A LOCAL PROJECTION, NOT UTM
The 90 trips are spread from roughly 8E/49N (Germany) to 55E/56N (Russia),
so there is no single UTM zone that covers the dataset.  Each trip is small
(a few km across), so a local equirectangular projection anchored at that
trip's own centroid is accurate to well under a metre across the trip extent
and is far cheaper than a proper reprojection - which matters, because this
is called millions of times during candidate generation.

Justify this choice in the report; the rubric asks for it explicitly.

COORDINATE ORDER WARNING
Every raw file is (longitude, latitude).  Folium wants [latitude, longitude].
Swapping these is the most common bug in this project.
"""

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def project(lon, lat, lon0: float, lat0: float):
    """Local equirectangular projection to metres about the anchor (lon0, lat0).

    Accepts scalars or arrays. Returns (x, y) in metres, east/north positive.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    coslat0 = np.cos(np.radians(lat0))
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * coslat0
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y


def trip_anchor(trip: dict) -> tuple[float, float]:
    """Centroid of the GPS track - the projection anchor for this trip."""
    return float(trip["track"][:, 0].mean()), float(trip["track"][:, 1].mean())


def project_trip(trip: dict):
    """Project a trip's track and nodes in one go.

    Returns (track_xy (N,2), node_xy (M,2), anchor (lon0, lat0)).
    Cache this per trip - do not recompute it inside loops.
    """
    lon0, lat0 = trip_anchor(trip)
    tx, ty = project(trip["track"][:, 0], trip["track"][:, 1], lon0, lat0)
    nx, ny = project(trip["nodes"][:, 0], trip["nodes"][:, 1], lon0, lat0)
    return np.column_stack([tx, ty]), np.column_stack([nx, ny]), (lon0, lat0)


def unproject(x, y, lon0: float, lat0: float):
    """Inverse of project(). Needed to put predicted geometry back on a map."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lon, lat


def point_to_segment(px, py, ax, ay, bx, by):
    """Perpendicular distance from point(s) P to segment(s) AB.

    Fully vectorised - broadcast P against many segments at once.
    Returns (dist, t) where t in [0,1] is the position of the foot of the
    perpendicular along AB (0 = at A, 1 = at B), clamped to the segment.
    """
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    ax, ay = np.asarray(ax, dtype=np.float64), np.asarray(ay, dtype=np.float64)
    bx, by = np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)

    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    # Zero-length arcs exist in this data; clamp t to 0 for them.
    t = np.where(denom > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(denom, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres. Use for GPS-to-GPS step distances."""
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(x1, y1, x2, y2):
    """Bearing in degrees [0,360) from (x1,y1) to (x2,y2) in PROJECTED metres.

    0 = north (+y), 90 = east (+x).
    """
    return np.degrees(np.arctan2(np.asarray(x2) - np.asarray(x1),
                                 np.asarray(y2) - np.asarray(y1))) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute angular difference, folded into [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def arc_endpoints_xy(trip: dict, node_xy: np.ndarray):
    """Projected endpoint coordinates of every arc.

    Returns (A (K,2), B (K,2)) aligned with arc_id = row index.
    """
    arcs = trip["arcs"]
    return node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]


def arc_lengths_m(trip: dict, node_xy: np.ndarray) -> np.ndarray:
    """Length in metres of every arc, indexed by arc_id."""
    A, B = arc_endpoints_xy(trip, node_xy)
    return np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
