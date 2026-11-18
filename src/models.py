"""Model zoo and the comparison harness.  OWNER: Rhea

Five models across five algorithm families.

SPLITTING: GroupKFold grouped by trip_id. A random split puts observations
from the same trip on both sides of a fold - they share a road network and
are seconds apart in time - which inflates every score. Explicitly penalised
by the rubric and the first thing a viva will probe.

PRIMARY METRIC: observation-level Top-1 accuracy. Row-level accuracy is
meaningless here: with ~7 candidates per observation, predicting "not the
arc" for everything already scores ~0.85. Top-1 asks the question the task
actually asks - of this observation's candidates, is the highest-scoring one
the true arc?
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss

from src.features import FEATURE_COLS, FEATURE_GROUPS, load_dataset

RANDOM_STATE = 42
N_SPLITS = 5

FAMILY = {
    "logistic_regression": "Linear / probabilistic",
    "knn": "Instance-based",
    "random_forest": "Ensemble (bagging)",
    "hist_gradient_boosting": "Boosting",
    "mlp": "Neural network",
}

CONFIG = {
    "logistic_regression": "C=1.0, class_weight=balanced, scaled",
    "knn": "k=25, weights=distance, scaled, fit on 50k subsample",
    "random_forest": "120 trees, min_samples_leaf=20, balanced_subsample",
    "hist_gradient_boosting": "max_iter=400, lr=0.05",
    "mlp": "(64,32), early_stopping, scaled",
}


def get_models() -> dict:
    """Scaling matters for 1, 2 and 5; tree models are scale-invariant."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        # 120 trees with a leaf floor of 20. At 1.2M rows, 300 unconstrained
        # trees took over 15 minutes per CV sweep for no measurable accuracy
        # gain - the dataset is large relative to its 15 features, so extra
        # trees mostly add cost. Documented as a deliberate budget choice.
        "random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                  max_iter=300, random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        ]),
    }


def _top1(df: pd.DataFrame, proba: np.ndarray) -> float:
    """Observation-level Top-1 accuracy: argmax candidate per observation."""
    tmp = pd.DataFrame({"trip_id": df.trip_id.to_numpy(),
                        "obs_idx": df.obs_idx.to_numpy(),
                        "proba": proba, "label": df.label.to_numpy()})
    best = tmp.loc[tmp.groupby(["trip_id", "obs_idx"]).proba.idxmax()]
    return float(best.label.mean())


def _continuity(df: pd.DataFrame, proba: np.ndarray, arcs_by_trip: dict) -> float:
    """Mean route continuity of the argmax prediction, across trips.

    Measures how often consecutive predicted arcs are actually connected in the
    road graph. The classifier sees each observation independently, so this is
    expected to be poor - that gap is exactly what the HMM exists to close.
    """
    from src.metrics import route_continuity
    tmp = pd.DataFrame({"trip_id": df.trip_id.to_numpy(),
                        "obs_idx": df.obs_idx.to_numpy(),
                        "arc_id": df.arc_id.to_numpy(), "proba": proba})
    best = tmp.loc[tmp.groupby(["trip_id", "obs_idx"]).proba.idxmax()]
    scores = []
    for tid, g in best.groupby("trip_id"):
        if tid in arcs_by_trip:
            g = g.sort_values("obs_idx")
            scores.append(route_continuity({"arcs": arcs_by_trip[tid]},
                                           g.arc_id.to_numpy()))
    return float(np.mean(scores)) if scores else float("nan")


# KNN stores the whole training set and computes distances at predict time, so
# it scales O(n) in memory and query cost. At 1.2M rows it is minutes per fold
# while every other model is seconds. We subsample its TRAINING data only -
# validation is still the full fold, so the reported numbers stay comparable.
# Document this in the report: it is a real limitation of the algorithm for
# this problem, not a measurement artefact, and it is part of why KNN is a
# poor fit here regardless of its accuracy.
FIT_SUBSAMPLE = {"knn": 50_000}


def _subsample(name, idx, rng):
    cap = FIT_SUBSAMPLE.get(name)
    if cap and len(idx) > cap:
        return rng.choice(idx, cap, replace=False)
    return idx


def cross_validate_model(name, model, df, feature_cols=FEATURE_COLS,
                         n_splits=N_SPLITS) -> dict:
    """Group-aware CV on the training partition."""
    X, y, groups = df[feature_cols], df.label, df.trip_id
    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(RANDOM_STATE)
    f1s, top1s, lls, fit_times = [], [], [], []

    for tr, va in gkf.split(X, y, groups):
        tr = _subsample(name, tr, rng)
        m = get_models()[name]
        t0 = time.time()
        m.fit(X.iloc[tr], y.iloc[tr])
        fit_times.append(time.time() - t0)
        p = m.predict_proba(X.iloc[va])[:, 1]
        f1s.append(f1_score(y.iloc[va], (p >= 0.5).astype(int),
                            average="macro", zero_division=0))
        top1s.append(_top1(df.iloc[va], p))
        lls.append(log_loss(y.iloc[va], p, labels=[0, 1]))

    return {
        "cv_macro_f1": np.mean(f1s), "cv_macro_f1_std": np.std(f1s),
        "cv_top1": np.mean(top1s), "cv_top1_std": np.std(top1s),
        "cv_log_loss": np.mean(lls), "train_time_s": np.mean(fit_times),
    }


def evaluate_on_student_test(model, df_test, feature_cols=FEATURE_COLS,
                             arcs_by_trip=None) -> dict:
    """Held-out student-test metrics. Never the evaluator partition."""
    t0 = time.time()
    p = model.predict_proba(df_test[feature_cols])[:, 1]
    infer_s = time.time() - t0
    pred = (p >= 0.5).astype(int)
    out = {
        "test_macro_f1": f1_score(df_test.label, pred, average="macro", zero_division=0),
        "test_top1": _top1(df_test, p),
        "test_log_loss": log_loss(df_test.label, p, labels=[0, 1]),
        "inference_time_s": infer_s,
        "inference_us_per_row": 1e6 * infer_s / len(df_test),
    }
    if arcs_by_trip:
        out["test_route_continuity"] = _continuity(df_test, p, arcs_by_trip)
    return out


def run_comparison(df_train, df_test, feature_cols=FEATURE_COLS,
                   arcs_by_trip=None,
                   out_csv="results/model_comparison.csv") -> pd.DataFrame:
    """Train and evaluate all five models. Writes the CSV, returns the table."""
    rows = []
    for name, model in get_models().items():
        print(f"\n=== {name} ===", flush=True)
        cv = cross_validate_model(name, model, df_train, feature_cols)
        print(f"  CV  macro_f1={cv['cv_macro_f1']:.4f}  top1={cv['cv_top1']:.4f}", flush=True)

        rng = np.random.default_rng(RANDOM_STATE)
        fit_idx = _subsample(name, np.arange(len(df_train)), rng)
        m = get_models()[name]
        m.fit(df_train[feature_cols].iloc[fit_idx], df_train.label.iloc[fit_idx])
        te = evaluate_on_student_test(m, df_test, feature_cols, arcs_by_trip)
        print(f"  TEST macro_f1={te['test_macro_f1']:.4f}  top1={te['test_top1']:.4f}  "
              f"logloss={te['test_log_loss']:.4f}", flush=True)

        rows.append({"model": name, "family": FAMILY[name],
                     "config": CONFIG[name], **cv, **te})

    df = pd.DataFrame(rows).sort_values("test_top1", ascending=False)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return df


def calibrate(model, df_cal, feature_cols=FEATURE_COLS, method="isotonic"):
    """Probability calibration. CORRECTNESS FIX, not just a novelty item.

    The classifier's output is used directly as the HMM emission probability,
    so a correct ranking is not sufficient - the numbers themselves must mean
    something. Two things here actively distort them: class_weight='balanced'
    deliberately reweights the classes, and tree ensembles are poorly
    calibrated by default.

    Fit on a split that is group-disjoint from both the training trips and the
    trips used for final reporting.
    """
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal.fit(df_cal[feature_cols], df_cal.label)
    return cal


def ablate_feature_groups(df_train, df_test, name="hist_gradient_boosting",
                          out_csv="results/ablation.csv") -> pd.DataFrame:
    """Drop one feature group at a time; record the Top-1 cost of each."""
    rows = []
    for drop, cols in [("none", [])] + list(FEATURE_GROUPS.items()):
        feats = [c for c in FEATURE_COLS if c not in cols]
        m = get_models()[name]
        m.fit(df_train[feats], df_train.label)
        p = m.predict_proba(df_test[feats])[:, 1]
        rows.append({"dropped_group": drop, "n_features": len(feats),
                     "test_top1": _top1(df_test, p)})
        print(f"  drop {drop:12} top1={rows[-1]['test_top1']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df["top1_delta"] = df.test_top1 - df.test_top1.iloc[0]
    df.to_csv(out_csv, index=False)
    return df


def export_final(model, feature_cols=FEATURE_COLS, path="artifacts/model.pkl",
                 extra: dict = None) -> None:
    joblib.dump({"model": model, "features": list(feature_cols),
                 "random_state": RANDOM_STATE, **(extra or {})}, path)


def load_final(path="artifacts/model.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    from src.io_utils import list_trips, load_trip

    print("loading datasets...", flush=True)
    tr = load_dataset("results/dataset_train.pkl")
    te = load_dataset("results/dataset_test.pkl")
    print(f"train {len(tr):,} rows / {tr.trip_id.nunique()} trips | "
          f"test {len(te):,} rows / {te.trip_id.nunique()} trips", flush=True)

    arcs_by_trip = {t: load_trip(f"data/student_test/{t}")["arcs"]
                    for t in list_trips("data/student_test")}

    table = run_comparison(tr, te, arcs_by_trip=arcs_by_trip)
    cols = ["model", "family", "cv_macro_f1", "cv_top1", "test_macro_f1",
            "test_top1", "test_log_loss", "test_route_continuity",
            "train_time_s", "inference_time_s"]
    print("\n" + table[cols].to_string(index=False))
    print(f"\nBEST: {table.iloc[0].model}  test_top1={table.iloc[0].test_top1:.4f}")
