"""Folium maps and matplotlib plots.  OWNER: A

Folium visualisation is MANDATORY. The report needs at least one map showing
GPS observations, road-network geometry, the reference route and the
predicted route, with a legend and layer controls.

COORDINATE ORDER: raw files are (lon, lat). Folium wants [lat, lon].
If your map lands in the Atlantic or Antarctica, this is why.
"""

import folium
import numpy as np

COLORS = {
    "gps": "#1f77b4",       # blue
    "network": "#999999",   # grey
    "reference": "#2ca02c", # green
    "predicted": "#d62728", # red
}


def trip_center(trip: dict) -> list:
    """[lat, lon] centre for folium.Map - note the order."""
    return [float(trip["track"][:, 1].mean()), float(trip["track"][:, 0].mean())]


def arc_latlon(trip: dict, arc_id: int) -> list:
    """[[lat, lon], [lat, lon]] endpoints of one arc, ready for folium."""
    a, b = trip["arcs"][arc_id]
    nodes = trip["nodes"]
    return [[float(nodes[a][1]), float(nodes[a][0])],
            [float(nodes[b][1]), float(nodes[b][0])]]


def arcs_to_latlon_path(trip: dict, arc_ids) -> list:
    """Flatten a sequence of arcs into a drawable [[lat, lon], ...] polyline.

    Deduplicate shared endpoints so the line does not double back on itself.
    """
    path = []
    for arc_id in arc_ids:
        if arc_id is None or arc_id < 0:
            continue
        for pt in arc_latlon(trip, int(arc_id)):
            if not path or path[-1] != pt:
                path.append(pt)
    return path


def plot_trip(trip: dict,
              pred_arcs=None,
              out_html: str = None,
              max_network_arcs: int = 3000) -> folium.Map:
    """Four-layer interactive map with LayerControl.

      1 raw GPS points      (blue circles)
      2 road network arcs   (thin grey) - subsampled on large trips
      3 reference route     (green) when trip['route'] is present
      4 predicted route     (red)   when pred_arcs is given

    Returns the map; also writes it if out_html is given.
    """
    raise NotImplementedError("A")


def plot_labels_debug(trip: dict, labels: np.ndarray, out_html: str = None):
    """GPS points coloured by assigned arc - the sanity check for labels.py.

    A correct alignment looks like a clean progression of colour bands along
    the road. Confetti means the DP is wrong.
    """
    raise NotImplementedError("A")
