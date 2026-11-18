"""HMM + Viterbi route reconstruction.  OWNER: Rhea

Newson & Krumm style map matching. Everything is in LOG SPACE - a 2000-step
product of small probabilities underflows float64 within a few hundred steps.

THE PROBLEM THIS SOLVES
The classifier scores each GPS observation independently, so consecutive
predictions can land on roads that do not touch. The vehicle appears to
teleport. Measured on student-test: Top-1 0.85 but route continuity 0.84,
meaning ~16% of road-to-road transitions are physically impossible.

HIDDEN STATES at step i = the candidate arcs for observation i.

EMISSION  log(classifier_proba)
    The trained classifier's probability that a candidate is the true arc IS
    the emission probability. This is the "learned emission" part - classical
    map matching uses a Gaussian on perpendicular distance instead.

TRANSITION  from candidate u at step i-1 to candidate v at step i:
    d_gps  = straight-line distance between observations i-1 and i
    d_road = shortest path through the road network from arc u to arc v
    logP   = -|d_gps - d_road| / beta

    If the vehicle really drove u -> v, the road distance should roughly
    match the GPS distance. A big mismatch means the move is implausible:
    either the roads are far apart in network terms, or they are adjacent in
    space but unreachable (opposite sides of a motorway).

DISCONNECTED: large finite penalty, never -inf. 89 of the 90 road networks in
this dataset are disconnected (median 19 components), so unreachable pairs
are common, not exceptional. A -inf would make entire paths infeasible and
collapse the decode.

GPS GAPS: with dt > gap_dt_s or d_gps > gap_dist_m we have no information
about what happened in between, so the transition relaxes toward uniform
rather than penalising a jump we cannot evaluate.
"""

import json
from heapq import heappush, heappop

import numpy as np
import pandas as pd

from src.geo import project_trip, arc_lengths_m, haversine_m

DEFAULT_CFG = {
    "beta": 20.0,                 # metres; transition decay scale
    "disconnect_penalty": -20.0,  # log-prob for unreachable candidate pairs
    "gap_dt_s": 30.0,
    "gap_dist_m": 500.0,
    "dijkstra_cutoff_mult": 3.0,
    "dijkstra_cutoff_base_m": 100.0,
    "beta_speed_coef": 0.0,
    "beta_gap_coef": 0.0,
    "dtw_frac_threshold": 0.10,
    "max_gap_s": 180.0,
}


class TripGraph:
    """Adjacency + cached shortest paths for one trip.

    Built once per decode. The cache matters enormously: a 2000-observation
    trip with 7 candidates each implies up to 98,000 transition queries, but
    the same (arc, arc) pairs recur constantly as the vehicle moves slowly
    relative to the sampling rate.
    """

    def __init__(self, trip: dict):
        self.arcs = trip["arcs"]
        _, node_xy, _ = project_trip(trip)
        self.arc_len = arc_lengths_m(trip, node_xy)

        # node -> list of (neighbour_node, length)
        self.adj = [[] for _ in range(len(trip["nodes"]))]
        for arc_id, (a, b) in enumerate(self.arcs):
            w = float(self.arc_len[arc_id])
            self.adj[a].append((b, w))
            self.adj[b].append((a, w))   # treat as undirected

        self._cache = {}

    def _sssp(self, src: int, cutoff: float) -> dict:
        """Single-source shortest paths from `src`, capped at `cutoff` metres.

        Cached per source node. A cached run computed with a cutoff at least
        as large as the one requested is reused, so in practice each source
        node is expanded once per trip.
        """
        cached = self._cache.get(src)
        if cached is not None and cached[0] >= cutoff:
            return cached[1]

        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heappop(heap)
            if d > cutoff:
                break
            if d > dist.get(n, float("inf")):
                continue
            for nb, w in self.adj[n]:
                nd = d + w
                if nd <= cutoff and nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    heappush(heap, (nd, nb))

        self._cache[src] = (cutoff, dist)
        return dist

    def travel_distance(self, u: int, t_u: float, v: int, t_v: float,
                        cutoff: float) -> float:
        """Distance the vehicle would actually travel from a point on arc u
        to a point on arc v.

        This is the quantity Newson & Krumm compare against the straight-line
        GPS step. Measuring endpoint-to-endpoint instead is wrong: two arcs
        that share a node are 0 m apart by that measure, yet the vehicle
        still moved the length of the GPS step, so every legitimate road
        change gets penalised and the decoder glues itself to one arc.

        Composed of three parts:
            distance remaining on u  +  network path  +  distance into v

        Both endpoints of u are candidate exits and both endpoints of v are
        candidate entries, because the graph is undirected and we do not know
        the direction of travel. The minimum over those four combinations is
        taken.
        """
        len_u = float(self.arc_len[u])
        len_v = float(self.arc_len[v])

        if u == v:
            return abs(t_v - t_u) * len_u

        u0, u1 = int(self.arcs[u][0]), int(self.arcs[u][1])
        v0, v1 = int(self.arcs[v][0]), int(self.arcs[v][1])

        # (exit node, distance still to travel along u to reach it)
        exits = ((u0, t_u * len_u), (u1, (1.0 - t_u) * len_u))
        entries = ((v0, t_v * len_v), (v1, (1.0 - t_v) * len_v))

        best = float("inf")
        for e_node, e_cost in exits:
            if e_cost > cutoff:
                continue
            dist = self._sssp(e_node, cutoff)
            for s_node, s_cost in entries:
                d = dist.get(s_node)
                if d is not None:
                    total = e_cost + d + s_cost
                    if total < best:
                        best = total
        return best


def build_graph(trip: dict) -> TripGraph:
    return TripGraph(trip)


def adaptive_beta(speed_mps: float, dt_s: float, cfg: dict) -> float:
    """Speed- and gap-aware transition scale.

    A beta tuned on slow urban driving is too strict at speed, where a
    legitimate 200 m step produces a large |d_gps - d_road| residual. Long
    sampling gaps deserve the same leniency. With both coefficients at 0 this
    reduces exactly to a fixed beta, so the ablation is a one-line change.
    """
    beta = cfg["beta"]
    scale = (1.0
             + cfg.get("beta_speed_coef", 0.0) * (speed_mps / 15.0)
             + cfg.get("beta_gap_coef", 0.0) * max(0.0, dt_s - 1.0) / 10.0)
    return beta * max(scale, 0.25)


def transition_logp(graph, u_arc, t_u, v_arc, t_v,
                    d_gps, dt_s, speed_mps, cfg) -> float:
    """Log transition probability between two candidate road positions."""
    # No information across a long gap - relax toward uniform rather than
    # penalising a move we cannot assess.
    if dt_s > cfg["gap_dt_s"] or d_gps > cfg["gap_dist_m"]:
        return -1.0

    cutoff = cfg["dijkstra_cutoff_mult"] * d_gps + cfg["dijkstra_cutoff_base_m"]
    d_road = graph.travel_distance(int(u_arc), float(t_u),
                                   int(v_arc), float(t_v), cutoff)
    if not np.isfinite(d_road):
        return cfg["disconnect_penalty"]

    beta = adaptive_beta(speed_mps, dt_s, cfg)
    return -abs(d_gps - d_road) / max(beta, 1e-6)


def viterbi(trip: dict, cands_with_proba: pd.DataFrame, cfg: dict = None) -> np.ndarray:
    """Decode the most likely arc sequence.

    cands_with_proba needs: obs_idx, arc_id, proba.
    Returns (N,) predicted arc_id per observation, -1 where no candidate.

    Standard O(N*k^2) forward pass with backpointers, then backtrack.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    graph = build_graph(trip)

    track = trip["track"]
    n_obs = len(track)
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]

    # Group candidates by observation. t_along locates the candidate WITHIN
    # its arc, which the travel-distance calculation needs.
    has_t = "t_along" in cands_with_proba.columns
    by_obs = {}
    for o, g in cands_with_proba.groupby("obs_idx"):
        t = g.t_along.to_numpy() if has_t else np.full(len(g), 0.5)
        by_obs[int(o)] = (g.arc_id.to_numpy(), g.proba.to_numpy(), t)
    steps = sorted(by_obs)
    if not steps:
        return np.full(n_obs, -1, dtype=np.int64)

    arcs0, proba0, t0 = by_obs[steps[0]]
    score = np.log(np.maximum(proba0, 1e-12))
    back = []           # per step: array mapping current candidate -> previous index
    prev_arcs, prev_t = arcs0, t0

    for j, i in enumerate(steps[1:], start=1):
        arcs_i, proba_i, t_i = by_obs[i]
        p_obs = steps[j - 1]

        d_gps = float(haversine_m(lon[p_obs], lat[p_obs], lon[i], lat[i]))
        dt_s = float(ts[i] - ts[p_obs])
        speed = d_gps / max(dt_s, 1e-6)

        emis = np.log(np.maximum(proba_i, 1e-12))
        new_score = np.empty(len(arcs_i))
        new_back = np.empty(len(arcs_i), dtype=np.int32)

        for c, v_arc in enumerate(arcs_i):
            best_val, best_k = -np.inf, 0
            for k, u_arc in enumerate(prev_arcs):
                val = score[k] + transition_logp(graph, u_arc, prev_t[k],
                                                 v_arc, t_i[c],
                                                 d_gps, dt_s, speed, cfg)
                if val > best_val:
                    best_val, best_k = val, k
            new_score[c] = best_val + emis[c]
            new_back[c] = best_k

        score, back = new_score, back + [new_back]
        prev_arcs, prev_t = arcs_i, t_i

    # Backtrack.
    out = np.full(n_obs, -1, dtype=np.int64)
    k = int(np.argmax(score))
    for idx in range(len(steps) - 1, -1, -1):
        obs = steps[idx]
        arcs_i = by_obs[obs][0]
        out[obs] = int(arcs_i[k])
        if idx > 0:
            k = int(back[idx - 1][k])
    return out


def classifier_only(cands_with_proba: pd.DataFrame) -> np.ndarray:
    """Baseline: independent argmax per observation, no sequential reasoning.

    This is what the HMM must beat. Report both.
    """
    idx = cands_with_proba.groupby("obs_idx")["proba"].idxmax()
    return cands_with_proba.loc[idx, "arc_id"].to_numpy()


def save_config(cfg: dict, path: str = "artifacts/hmm_config.json") -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_config(path: str = "artifacts/hmm_config.json") -> dict:
    try:
        with open(path) as f:
            return {**DEFAULT_CFG, **json.load(f)}
    except FileNotFoundError:
        return dict(DEFAULT_CFG)
