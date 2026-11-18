"""Run a subset of models and append to the comparison CSV.  OWNER: Rhea

    python -m src.run_models hist_gradient_boosting mlp

Exists because the five models differ in cost by two orders of magnitude on
1.2M rows: HistGradientBoosting cross-validates in under a minute while
RandomForest with 300 trees takes ~15. Running them as separate processes
means a usable comparison table exists early instead of after the slowest
model finishes.

Each run appends to results/model_comparison.csv, de-duplicating on model
name so a re-run replaces that model's row rather than duplicating it.
"""

import os
import sys

import pandas as pd

from src.features import FEATURE_COLS, load_dataset
from src.io_utils import list_trips, load_trip
from src.models import (get_models, cross_validate_model,
                        evaluate_on_student_test, _subsample,
                        FAMILY, CONFIG, RANDOM_STATE)

OUT_CSV = "results/model_comparison.csv"


def run(names, df_train, df_test, arcs_by_trip):
    import numpy as np
    rows = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        cv = cross_validate_model(name, get_models()[name], df_train)
        print(f"  CV  macro_f1={cv['cv_macro_f1']:.4f}  top1={cv['cv_top1']:.4f}", flush=True)

        rng = np.random.default_rng(RANDOM_STATE)
        idx = _subsample(name, np.arange(len(df_train)), rng)
        m = get_models()[name]
        m.fit(df_train[FEATURE_COLS].iloc[idx], df_train.label.iloc[idx])
        te = evaluate_on_student_test(m, df_test, FEATURE_COLS, arcs_by_trip)
        print(f"  TEST macro_f1={te['test_macro_f1']:.4f}  top1={te['test_top1']:.4f}  "
              f"logloss={te['test_log_loss']:.4f}  "
              f"continuity={te.get('test_route_continuity', float('nan')):.4f}", flush=True)

        rows.append({"model": name, "family": FAMILY[name],
                     "config": CONFIG[name], **cv, **te})
    return pd.DataFrame(rows)


def append_results(new: pd.DataFrame, out_csv=OUT_CSV):
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        old = old[~old.model.isin(new.model)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values("test_top1", ascending=False)
    new.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(new)} models)")
    return new


if __name__ == "__main__":
    names = sys.argv[1:] or ["hist_gradient_boosting"]
    print(f"running: {names}", flush=True)

    df_train = load_dataset("results/dataset_train.pkl")
    df_test = load_dataset("results/dataset_test.pkl")
    print(f"train {len(df_train):,} rows | test {len(df_test):,} rows", flush=True)

    arcs_by_trip = {t: load_trip(f"data/student_test/{t}")["arcs"]
                    for t in list_trips("data/student_test")}

    table = append_results(run(names, df_train, df_test, arcs_by_trip))
    cols = ["model", "family", "cv_macro_f1", "cv_top1", "test_top1",
            "test_macro_f1", "test_log_loss", "test_route_continuity"]
    print("\n" + table[[c for c in cols if c in table]].to_string(index=False))
    print(f"\nBEST SO FAR: {table.iloc[0].model}  test_top1={table.iloc[0].test_top1:.4f}")
