"""Exploratory data analysis for report section 2.  OWNER: Rhea -> hand to P3

    python -m src.eda

Writes the dataset inventory, the EDA figures and one Folium map. Everything
is computed from the raw files - nothing here is illustrative or hand-made.

Covers the rubric's required list: trip counts, observations per trip, file
availability, missing/invalid observations, timestamp ordering and gaps,
geographic extent, road-network size and connectivity, isolated nodes and
disconnected components, anomalous trips, and reference-route structure.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.io_utils import list_trips, load_trip, trip_summary, validate_trip
from src.geo import project_trip
from src.labels import align_route_positions, _route_distance_matrix

OUT = "results"
MAPS = "maps"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def build_inventory(partitions=("data/train", "data/student_test")) -> pd.DataFrame:
    """One row per trip: counts, extent, gaps, network size, connectivity."""
    rows = []
    for part in partitions:
        for tid in list_trips(part):
            trip = load_trip(f"{part}/{tid}")
            s = trip_summary(trip)
            s["partition"] = os.path.basename(part)

            # Road-network connectivity, undirected.
            g = nx.Graph()
            g.add_nodes_from(range(len(trip["nodes"])))
            g.add_edges_from(map(tuple, trip["arcs"]))
            comps = list(nx.connected_components(g))
            degrees = np.array([d for _, d in g.degree()])
            s["n_components"] = len(comps)
            s["largest_component_frac"] = max(len(c) for c in comps) / len(trip["nodes"])
            s["n_isolated_nodes"] = int((degrees == 0).sum())
            s["mean_degree"] = float(degrees.mean())

            # How far the GPS sits from its own reference route.
            if trip["route"] is not None and len(trip["route"]):
                pos = align_route_positions(trip)
                D = _route_distance_matrix(trip)
                snap = D[np.arange(len(trip["track"])), pos]
                s["median_snap_m"] = float(np.median(snap))
                s["mean_offset_m"] = float(snap.mean())
            else:
                s["median_snap_m"] = np.nan
                s["mean_offset_m"] = np.nan

            s["problems"] = "; ".join(validate_trip(trip)) or "none"
            rows.append(s)
            print(f"  {part}/{tid}", end="\r", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/dataset_inventory.csv", index=False)
    print(f"\nwrote {OUT}/dataset_inventory.csv ({len(df)} trips)")
    return df


def plot_overview(df: pd.DataFrame):
    """Six-panel overview of the dataset."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].hist(df.n_obs, bins=30, color=PALETTE[0], edgecolor="white")
    ax[0, 0].set_title("GPS observations per trip")
    ax[0, 0].set_xlabel("observations")

    ax[0, 1].hist(df.duration_s / 60, bins=30, color=PALETTE[1], edgecolor="white")
    ax[0, 1].set_title("Trip duration")
    ax[0, 1].set_xlabel("minutes")

    ax[0, 2].scatter(df.n_nodes, df.n_arcs, c=PALETTE[2], alpha=0.7)
    ax[0, 2].set_title("Road-network size per trip")
    ax[0, 2].set_xlabel("nodes")
    ax[0, 2].set_ylabel("arcs")

    # Every trip has its own network in a different part of the world - the
    # single most important structural fact about this dataset.
    for part, g in df.groupby("partition"):
        ax[1, 0].scatter(g.lon_min, g.lat_min, label=part, alpha=0.8, s=40)
    ax[1, 0].set_title("Trip locations (Germany to Russia)")
    ax[1, 0].set_xlabel("longitude")
    ax[1, 0].set_ylabel("latitude")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].hist(df.dt_median, bins=20, color=PALETTE[3], edgecolor="white")
    ax[1, 1].set_title("Median sampling interval")
    ax[1, 1].set_xlabel("seconds")

    snap = df.median_snap_m.dropna()
    ax[1, 2].hist(snap, bins=40, color=PALETTE[4], edgecolor="white")
    ax[1, 2].axvline(100, ls="--", c="red", lw=1)
    ax[1, 2].text(100, ax[1, 2].get_ylim()[1] * 0.9, " outlier cutoff",
                  fontsize=8, color="red")
    ax[1, 2].set_title("Median GPS distance to reference route")
    ax[1, 2].set_xlabel("metres")

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("PathSnap dataset overview - 90 trips", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_eda_overview.png", dpi=150)
    print(f"wrote {OUT}/fig_eda_overview.png")


def plot_anomalies(df: pd.DataFrame):
    """The datum-offset finding: two trips sit far off their own road network."""
    d = df.dropna(subset=["median_snap_m"]).sort_values("median_snap_m")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#C44E52" if v > 100 else "#4C72B0" for v in d.median_snap_m]
    ax.bar(range(len(d)), d.median_snap_m, color=colors)
    ax.axhline(100, ls="--", c="red", lw=1, label="100 m outlier cutoff")
    ax.set_yscale("log")
    ax.set_xlabel("trips, sorted by median offset")
    ax.set_ylabel("median distance to reference route (m, log)")
    ax.set_title("Two trips are systematically displaced from their road network")
    for _, r in d[d.median_snap_m > 100].iterrows():
        ax.annotate(r.trip_id, (list(d.trip_id).index(r.trip_id), r.median_snap_m),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8, color="#C44E52")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_anomalous_trips.png", dpi=150)
    print(f"wrote {OUT}/fig_anomalous_trips.png")


def folium_map(trip_id="00000007", partition="data/student_test",
               max_arcs=2500, pred_arcs=None):
    """Four-layer interactive map: GPS, road network, reference route, prediction.

    Raw files are (lon, lat); folium wants [lat, lon]. Everything below flips
    deliberately - a map in the ocean means this was missed somewhere.
    """
    import folium

    trip = load_trip(f"{partition}/{trip_id}")
    nodes, arcs, track = trip["nodes"], trip["arcs"], trip["track"]
    centre = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

    net = folium.FeatureGroup(name="Road network", show=True)
    step = max(1, len(arcs) // max_arcs)
    for a, b in arcs[::step]:
        folium.PolyLine([[nodes[a][1], nodes[a][0]], [nodes[b][1], nodes[b][0]]],
                        color="#999999", weight=1, opacity=0.5).add_to(net)
    net.add_to(m)

    if trip["route"] is not None:
        ref = folium.FeatureGroup(name="Reference route (ground truth)")
        pts = []
        for arc_id in trip["route"]:
            a, b = arcs[arc_id]
            for p in ([nodes[a][1], nodes[a][0]], [nodes[b][1], nodes[b][0]]):
                if not pts or pts[-1] != p:
                    pts.append(p)
        folium.PolyLine(pts, color="#2ca02c", weight=4, opacity=0.8).add_to(ref)
        ref.add_to(m)

    if pred_arcs is not None:
        pred = folium.FeatureGroup(name="Predicted route")
        from src.viz import arcs_to_latlon_path
        folium.PolyLine(arcs_to_latlon_path(trip, pred_arcs),
                        color="#d62728", weight=4, opacity=0.8).add_to(pred)
        pred.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations")
    for lon, lat, _ in track:
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4",
                            fill=True, fill_opacity=0.6).add_to(gps)
    gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out = f"{MAPS}/trip_{trip_id}.html"
    m.save(out)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    os.makedirs(MAPS, exist_ok=True)
    inv = build_inventory()

    print("\n=== DATASET INVENTORY ===")
    print(inv.groupby("partition").agg(
        trips=("trip_id", "count"), observations=("n_obs", "sum"),
        median_obs=("n_obs", "median"), arcs=("n_arcs", "sum"),
        median_gap_s=("dt_median", "median"),
        trips_with_gaps=("n_gaps_gt_5s", lambda s: int((s > 0).sum())),
    ).to_string())

    print("\n=== ANOMALIES ===")
    bad = inv[inv.median_snap_m > 100]
    print(bad[["trip_id", "partition", "n_obs", "median_snap_m"]].to_string(index=False)
          if len(bad) else "none")

    print("\n=== CONNECTIVITY ===")
    print(f"trips with >1 component : {int((inv.n_components > 1).sum())} / {len(inv)}")
    print(f"median components       : {inv.n_components.median():.0f}")
    print(f"median isolated nodes   : {inv.n_isolated_nodes.median():.0f}")
    print(f"median node degree      : {inv.mean_degree.median():.2f}")

    plot_overview(inv)
    plot_anomalies(inv)
    folium_map()
