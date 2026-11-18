"""Bayesian hyperparameter optimisation.  OWNER: C

MANDATORY deliverable - Bayesian search specifically, not grid or random.
The search RECORD must be retained, so always write study.trials_dataframe()
to results/optuna_trials.csv.

Objective: group-aware CV Top-1 accuracy (or macro F1 - pick one and justify
it in the report). Top-1 is the better choice because it is the metric the
actual task cares about.

The evaluator partition must not influence the search space, the objective,
the hyperparameters, the model, or any threshold.
"""

import optuna
import pandas as pd

N_TRIALS = 40
RANDOM_STATE = 42


def suggest_hgb(trial) -> dict:
    """Search space for HistGradientBoosting. Justify each range in the report."""
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }


def suggest_rf(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def objective(trial, df_train, feature_cols, model_name="hist_gradient_boosting"):
    """Group-aware CV Top-1 accuracy for one trial's parameters."""
    raise NotImplementedError("C")


def run_study(df_train, feature_cols, n_trials=N_TRIALS,
              out_csv="results/optuna_trials.csv"):
    """Run the search, persist the trial record, return the study.

    sampler=TPESampler is what makes this Bayesian - state that in the report.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    raise NotImplementedError("C: call study.optimize(...) then write out_csv")


def plot_study(study, out_dir="results") -> None:
    """Optimisation history + parameter importances. Both go in the report."""
    raise NotImplementedError("C")
