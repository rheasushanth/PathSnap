"""Model zoo and the comparison harness.  OWNER: C

Five models, deliberately spread across algorithm families - the rubric
rewards the spread, not five flavours of boosting.

SPLITTING: GroupKFold grouped by trip_id. A random split puts observations
from the same trip on both sides of the fold and inflates every score
enormously. This is explicitly penalised and the viva will ask about it.

Fit the scaler INSIDE each fold, on train only.
Set random_state=42 everywhere and record it.
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

RANDOM_STATE = 42
N_SPLITS = 5


def get_models() -> dict:
    """The five candidate models. Scaling matters for 1, 2 and 5; not for 3, 4."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                  max_iter=300, random_state=RANDOM_STATE)),
        ]),
    }


def cross_validate_model(name, model, df, feature_cols) -> dict:
    """Group-aware CV on the training partition.

    Must return, for results/model_comparison.csv:
      model, family, key_config, cv_macro_f1, cv_macro_f1_std, cv_top1,
      log_loss, train_time_s, inference_time_s
    """
    raise NotImplementedError("C")


def evaluate_on_student_test(name, model, df_test, feature_cols) -> dict:
    """Student-test metrics for the comparison table. Never the evaluator partition."""
    raise NotImplementedError("C")


def run_comparison(df_train, df_test, feature_cols,
                   out_csv="results/model_comparison.csv") -> pd.DataFrame:
    """Train and evaluate all five, write the CSV, return the table."""
    raise NotImplementedError("C")


def calibrate(model, df_cal, feature_cols, method="isotonic"):
    """Probability calibration. NOVELTY + CORRECTNESS - do not skip this.

    The classifier's output is used directly as the HMM EMISSION probability,
    so the ranking being right is not enough: the numbers themselves have to
    mean something. Two things in this pipeline actively distort them:

      1. class_weight='balanced' (used in 3 of the 5 models) deliberately
         reweights the classes and inflates positive-class probabilities.
      2. Tree ensembles are poorly calibrated by default - RandomForest
         averages votes and pushes probabilities toward the middle.

    Miscalibrated emissions silently reweight the HMM against the transition
    term, so Viterbi balances the wrong two quantities.

    Fit on a held-out calibration split that is GROUP-DISJOINT from both the
    training trips and the trips used for final reporting.

    Report: reliability diagram before/after, Brier score before/after, and
    Top-1 accuracy after Viterbi with calibrated vs uncalibrated emissions.
    That last number is the one that matters.
    """
    from sklearn.calibration import CalibratedClassifierCV
    raise NotImplementedError("C: CalibratedClassifierCV(model, method=method, cv='prefit')")


def ablate_feature_groups(df_train, df_test, feature_cols,
                          out_csv="results/ablation.csv"):
    """Feature-group ablation. Cheap analytical depth for the report.

    Groups: geometric / directional / kinematic / topological (see
    features.FEATURE_COLS). Drop one group at a time, retrain the chosen
    model, record the Top-1 drop. Shows WHICH signals carry the task rather
    than just that the model works.
    """
    raise NotImplementedError("C")


def export_final(model, scaler, feature_cols, path="artifacts/model.pkl") -> None:
    """Freeze everything inference needs into one file for the API."""
    joblib.dump({"model": model, "scaler": scaler, "features": list(feature_cols),
                 "random_state": RANDOM_STATE}, path)


def load_final(path="artifacts/model.pkl"):
    return joblib.load(path)
