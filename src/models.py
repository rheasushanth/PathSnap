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


def export_final(model, scaler, feature_cols, path="artifacts/model.pkl") -> None:
    """Freeze everything inference needs into one file for the API."""
    joblib.dump({"model": model, "scaler": scaler, "features": list(feature_cols),
                 "random_state": RANDOM_STATE}, path)


def load_final(path="artifacts/model.pkl"):
    return joblib.load(path)
