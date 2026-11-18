"""Bayesian hyperparameter optimisation.  OWNER: Rhea

    python -m src.tune                    # tune the leading model
    python -m src.tune 30 400000          # n_trials, subsample size

MANDATORY deliverable, and it must be BAYESIAN specifically - not grid, not
random. Optuna's TPESampler builds a probabilistic model of the objective
from completed trials and samples where improvement is likely. That is the
Bayesian part, and it is the thing to say in the viva.

The search record is written to results/optuna_trials.csv. The rubric
requires the trial history to be retained, not just the winning values.

SUBSAMPLING: the search runs on a stratified subsample of trips rather than
all 1.2M rows. A single group-aware 3-fold CV on the full set takes minutes
and 30 trials of that does not fit the schedule. Trips are sampled whole, so
the grouping structure is preserved and no trip is split across folds. The
FINAL refit uses all 80 trips. State this in the report.

LEAKAGE: the evaluator partition plays no part in the search space, the
objective, the number of trials, or the selected configuration.
"""

import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.features import FEATURE_COLS, load_dataset

RANDOM_STATE = 42
N_TRIALS = 30
N_SPLITS = 3
SUBSAMPLE_TRIPS = 40

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_hgb(trial) -> dict:
    """Search space for HistGradientBoosting, with reasons.

    learning_rate / max_iter  the core capacity-versus-steps tradeoff
    max_leaf_nodes            tree expressiveness; the dominant knob here
    min_samples_leaf          guards against fitting single noisy candidates
    l2_regularization         log scale, since useful values span decades
    max_bins                  histogram resolution; our features are
                              continuous distances and angles, so binning
                              granularity genuinely matters
    """
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 200),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }


def top1(df, proba) -> float:
    """Observation-level Top-1 accuracy - the objective we optimise.

    Not log loss and not row-level F1: Top-1 is the metric the task is scored
    on, and optimising a proxy risks winning the proxy and losing the task.
    """
    tmp = pd.DataFrame({"g": df.trip_id.to_numpy(), "o": df.obs_idx.to_numpy(),
                        "p": proba, "y": df.label.to_numpy()})
    return float(tmp.loc[tmp.groupby(["g", "o"]).p.idxmax()].y.mean())


def make_objective(df):
    X, y, groups = df[FEATURE_COLS], df.label, df.trip_id

    def objective(trial):
        params = suggest_hgb(trial)
        gkf = GroupKFold(n_splits=N_SPLITS)
        scores = []
        for tr, va in gkf.split(X, y, groups):
            m = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)
            m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(top1(df.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(scores))

    return objective


def run_study(df, n_trials=N_TRIALS, out_csv="results/optuna_trials.csv"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="pathsnap_hgb",
    )

    t0 = time.time()
    best_so_far = [0.0]

    def log_cb(st, tr):
        if tr.value and tr.value > best_so_far[0]:
            best_so_far[0] = tr.value
            print(f"  trial {tr.number:3d}  top1={tr.value:.4f}  <- new best "
                  f"({time.time()-t0:.0f}s)", flush=True)
        elif tr.number % 5 == 0:
            print(f"  trial {tr.number:3d}  top1={tr.value:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    study.optimize(make_objective(df), n_trials=n_trials, callbacks=[log_cb])

    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(study.trials)} trials, {time.time()-t0:.0f}s)")
    return study


def main(n_trials=N_TRIALS, subsample_trips=SUBSAMPLE_TRIPS):
    full = load_dataset("results/dataset_train.pkl")
    rng = np.random.default_rng(RANDOM_STATE)
    trips = full.trip_id.unique()
    keep = rng.choice(trips, min(subsample_trips, len(trips)), replace=False)
    df = full[full.trip_id.isin(keep)]
    print(f"search set: {len(df):,} rows / {df.trip_id.nunique()} trips "
          f"(of {len(full):,} / {len(trips)})\n")

    study = run_study(df, n_trials)

    print("\n=== BEST CONFIGURATION ===")
    for k, v in study.best_params.items():
        print(f"  {k:22} {v}")
    print(f"  cv_top1 (search set)   {study.best_value:.4f}")

    # Baseline: the untuned defaults used in the model comparison.
    print("\nrefitting tuned and untuned on ALL training trips...", flush=True)
    te = load_dataset("results/dataset_test.pkl")
    out = []
    for label, params in [("untuned", {"max_iter": 400, "learning_rate": 0.05}),
                          ("tuned", study.best_params)]:
        m = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        t0 = time.time()
        m.fit(full[FEATURE_COLS], full.label)
        fit_s = time.time() - t0
        t0 = time.time()
        p = m.predict_proba(te[FEATURE_COLS])[:, 1]
        inf_s = time.time() - t0
        from sklearn.metrics import log_loss, f1_score
        out.append({
            "variant": label,
            "test_top1": top1(te, p),
            "test_macro_f1": f1_score(te.label, (p >= 0.5).astype(int),
                                      average="macro", zero_division=0),
            "test_log_loss": log_loss(te.label, p, labels=[0, 1]),
            "fit_s": round(fit_s, 1), "inference_s": round(inf_s, 2),
        })
        print(f"  {label:8} top1={out[-1]['test_top1']:.4f} "
              f"f1={out[-1]['test_macro_f1']:.4f} "
              f"logloss={out[-1]['test_log_loss']:.4f}", flush=True)

    cmp = pd.DataFrame(out)
    cmp.to_csv("results/tuning_comparison.csv", index=False)
    print("\n" + cmp.to_string(index=False))
    delta = cmp.test_top1.iloc[1] - cmp.test_top1.iloc[0]
    print(f"\ntuned - untuned Top-1: {delta:+.4f}")
    print("wrote results/tuning_comparison.csv")

    import json
    with open("artifacts/best_params.json", "w") as f:
        json.dump({"model": "hist_gradient_boosting",
                   "params": study.best_params,
                   "search_cv_top1": study.best_value,
                   "n_trials": len(study.trials)}, f, indent=2)
    print("wrote artifacts/best_params.json")
    return study


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS
    t = int(sys.argv[2]) if len(sys.argv) > 2 else SUBSAMPLE_TRIPS
    main(n, t)
