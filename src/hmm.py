"""HMM + Viterbi route reconstruction.  OWNER: B

Newson & Krumm style map matching. Work in LOG SPACE throughout.

HIDDEN STATES at step i = the candidate arcs for observation i.

EMISSION  log(classifier_proba + 1e-12)
    The tuned classifier's probability that a candidate is correct IS the
    emission probability. Say exactly this in report section 5 - it is one of
    the questions asked.

TRANSITION  from candidate u at step i-1 to candidate v at step i:
    d_gps  = haversine distance between observations i-1 and i
    d_road = network shortest-path distance between arc u and arc v
    delta  = |d_gps - d_road|
    logP   = -delta / BETA
    Intuition: if the vehicle really travelled u -> v, the road distance
    should match the straight-line GPS distance. Large mismatch = implausible.

    Compute d_road with a Dijkstra CUTOFF (3*d_gps + 100 m) on a node graph
    weighted by arc length. Never all-pairs - far too slow. Cache per (u,v).

DISCONNECTED: large finite penalty, NEVER -inf. A -inf row can make the whole
path infeasible and the decode collapses.

GPS GAPS: when dt > GAP_DT_S or d_gps > GAP_DIST_M we have no information about
what happened in between, so soften the transition toward uniform rather than
penalising a jump we cannot evaluate.

Tune BETA and DISCONNECT_PENALTY on training + student_test ONLY.
Save the chosen values to artifacts/hmm_config.json.
"""

import json
import numpy as np
import pandas as pd

DEFAULT_CFG = {
    "beta": 20.0,                 # metres; transition decay scale
    "disconnect_penalty": -20.0,  # log-prob for unreachable candidate pairs
    "gap_dt_s": 30.0,
    "gap_dist_m": 500.0,
    "dijkstra_cutoff_mult": 3.0,
    "dijkstra_cutoff_base_m": 100.0,
    # adaptive transition scale (see adaptive_beta); set to 0 for fixed beta
    "beta_speed_coef": 0.0,
    "beta_gap_coef": 0.0,
    # confidence thresholds (see confidence_label)
    "dtw_frac_threshold": 0.10,
    "max_gap_s": 180.0,
}


def build_graph(trip: dict):
    """NetworkX DiGraph over nodes, edges weighted by arc length in metres.

    Keep the arc_id on each edge - the decoder needs to map paths back to arcs.
    """
    raise NotImplementedError("B")


def transition_logp(trip, graph, u_arc, v_arc, d_gps, cfg) -> float:
    """Log transition probability between two candidate arcs. See module docstring."""
    raise NotImplementedError("B")


def viterbi(trip: dict, cands_with_proba: pd.DataFrame, cfg: dict = None) -> np.ndarray:
    """Decode the most likely arc sequence.

    cands_with_proba: obs_idx, arc_id, proba (+ feature columns, ignored here)
    Returns (N,) predicted arc_id per observation.

    Standard O(N*k^2) forward pass with backpointers, then backtrack.
    Observations with no candidates must not occur - candidates.py guarantees
    at least one.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    raise NotImplementedError("B")


def adaptive_beta(speed_mps: float, dt_s: float, cfg: dict) -> float:
    """Speed- and gap-aware transition scale. NOVELTY - a learned-ish transition.

    A fixed beta is the textbook choice and it is wrong in two regimes:
      - fast motion: a legitimate 200 m step looks like a huge |d_gps - d_road|
        mismatch under a beta tuned on slow urban driving
      - long sampling gaps: the vehicle genuinely could have gone further, so
        the model should be MORE permissive, not less

    beta_eff = beta * (1 + a*speed_norm + b*gap_norm)

    Ablate it: fixed beta vs adaptive beta, same decoder, same trips. Report
    the delta. This is one of the few places the design departs from the
    standard Newson & Krumm formulation, so it is worth defending properly.
    """
    raise NotImplementedError("B")


def dtw_deviation(trip: dict, pred_arc_ids: np.ndarray) -> float:
    """Mean DTW deviation (metres) between the GPS trace and the decoded route.

    Dynamic Time Warping aligns two sequences of DIFFERENT lengths - here the
    N GPS observations against the decoded route polyline - without assuming
    they are sampled in step. Plain pointwise distance cannot do this.

    Use it as a per-trip CONFIDENCE signal, not a loss. If mean deviation
    exceeds ~10% of path length, the decode has probably gone somewhere the
    vehicle did not.
    """
    raise NotImplementedError("B")


def confidence_label(trip: dict, pred_arc_ids: np.ndarray, cands: pd.DataFrame,
                     cfg: dict = None) -> dict:
    """High / Medium / Low confidence for one decoded trip. NOVELTY.

    Nothing else in the system tells the evaluator WHEN TO DISTRUST the output.
    Four independent signals, each catching a different failure mode:

      1. mean emission probability of the chosen path (model certainty)
      2. DTW deviation vs path length           (geometric plausibility)
      3. largest timestamp gap in the trip      (coverage)
      4. fraction of transitions that were disconnected-penalised (continuity)

    Returns {label, signals: {...}, fallback_used: bool}.

    FALLBACK RULE, straight from the Zepto pipeline: on Low confidence, fall
    back to the per-observation nearest-arc snap rather than the decoded route.
    A bad decode must never make the output worse than the trivial baseline.
    Surface the label in the UI - it demonstrates engineering maturity and it
    is a good viva answer.
    """
    raise NotImplementedError("B")


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
