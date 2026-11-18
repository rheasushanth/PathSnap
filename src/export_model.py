"""Train the chosen model on all training trips and export it for the API.

    python -m src.export_model logistic_regression

Writes artifacts/model.pkl containing the fitted estimator, the exact feature
list it was trained on, and the candidate-generation settings. The API must
regenerate candidates with the SAME radius and k it was trained with, so
those travel with the model rather than being duplicated as constants in the
API.
"""

import sys
import time

import numpy as np

from src.features import FEATURE_COLS, load_dataset
from src.models import get_models, export_final, evaluate_on_student_test, RANDOM_STATE

RADIUS_M = 50.0
K = 10


def main(name="logistic_regression"):
    print(f"training {name} on the full training partition", flush=True)
    tr = load_dataset("results/dataset_train.pkl")
    te = load_dataset("results/dataset_test.pkl")
    print(f"  train {len(tr):,} rows / {tr.trip_id.nunique()} trips", flush=True)

    model = get_models()[name]
    t0 = time.time()
    model.fit(tr[FEATURE_COLS], tr.label)
    fit_s = time.time() - t0
    print(f"  fitted in {fit_s:.1f}s", flush=True)

    metrics = evaluate_on_student_test(model, te, FEATURE_COLS)
    print(f"  student-test Top-1     {metrics['test_top1']:.4f}", flush=True)
    print(f"  student-test Macro F1  {metrics['test_macro_f1']:.4f}", flush=True)
    print(f"  student-test Log Loss  {metrics['test_log_loss']:.4f}", flush=True)

    export_final(model, FEATURE_COLS, "artifacts/model.pkl", extra={
        "model_name": name,
        "radius_m": RADIUS_M,
        "k": K,
        "trained_rows": int(len(tr)),
        "trained_trips": int(tr.trip_id.nunique()),
        "student_test_top1": float(metrics["test_top1"]),
        "student_test_macro_f1": float(metrics["test_macro_f1"]),
        "fit_seconds": round(fit_s, 1),
    })
    print("  wrote artifacts/model.pkl", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logistic_regression")
