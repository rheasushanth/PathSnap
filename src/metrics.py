"""Evaluation metrics.  OWNER: C

Two levels of metric, and they are NOT the same thing:

  row level  - binary classification over (obs, candidate) pairs. With k=10
               candidates roughly 1 row in 10 is positive, so plain accuracy
               sits near 0.9 while predicting all-negative. Always report
               MACRO F1, never bare accuracy.

  obs level  - Top-1 accuracy: group by (trip_id, obs_idx), take the argmax
               candidate, check its arc_id against the true arc. THIS is the
               metric that reflects the actual task. Make it the primary
               model-selection metric and justify that in the report.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)


def evaluate(y_true, y_pred, proba=None) -> dict:
    """Row-level binary classification metrics."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if proba is not None:
        out["log_loss"] = log_loss(y_true, proba, labels=[0, 1])
    return out


def top1_accuracy(df: pd.DataFrame,
                  group_cols=("trip_id", "obs_idx"),
                  proba_col="proba",
                  label_col="label") -> float:
    """Fraction of observations whose highest-scoring candidate is the true arc.

    df needs: group_cols, proba_col, label_col (1 on the true candidate).
    """
    group_cols = list(group_cols)
    best = df.loc[df.groupby(group_cols)[proba_col].idxmax()]
    return float(best[label_col].mean())


def candidate_miss_rate(df: pd.DataFrame,
                        group_cols=("trip_id", "obs_idx"),
                        label_col="label") -> float:
    """Fraction of observations where the true arc was never in the candidate set.

    A hard ceiling on Top-1 accuracy. Tune radius/k against this.
    """
    group_cols = list(group_cols)
    has_true = df.groupby(group_cols)[label_col].max()
    return float((has_true == 0).mean())


def route_continuity(trip: dict, pred_arc_ids: np.ndarray) -> float:
    """Fraction of consecutive predicted-arc CHANGES that are topologically valid.

    A transition is valid when the two arcs share a node (or are the same arc).
    Repeats of the same arc are excluded from the denominator - a stationary
    vehicle should not inflate the score.

    This is the metric that shows what the HMM bought you: classifier-only
    predictions typically score well below the Viterbi-decoded route.
    """
    arcs = trip["arcs"]
    pred = np.asarray(pred_arc_ids)
    changes = [(a, b) for a, b in zip(pred[:-1], pred[1:]) if a != b and a >= 0 and b >= 0]
    if not changes:
        return 1.0
    valid = sum(1 for a, b in changes if set(arcs[a]) & set(arcs[b]))
    return valid / len(changes)


def sequence_metrics(trip: dict, pred_arc_ids, true_arc_ids) -> dict:
    """Observation-level agreement plus continuity, for the HMM comparison table."""
    pred, true = np.asarray(pred_arc_ids), np.asarray(true_arc_ids)
    mask = true >= 0
    return {
        "pointwise_accuracy": float((pred[mask] == true[mask]).mean()) if mask.any() else 0.0,
        "route_continuity": route_continuity(trip, pred),
        "n_evaluated": int(mask.sum()),
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
