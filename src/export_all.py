"""Export every trained model so the API can serve any of them.

    python -m src.export_all                      # all models
    python -m src.export_all hist_gradient_boosting

Writes artifacts/model_<name>.pkl per model plus artifacts/models.json
describing what is available. The API reads that index and lets the
evaluator choose, which turns the model comparison from a table in the
report into something they can drive themselves.

artifacts/model.pkl stays as the DEFAULT the API loads at startup, and
points at whichever model is currently best.
"""

import json
import os
import shutil
import sys
import time

from sklearn.metrics import f1_score, log_loss

from src.features import FEATURE_COLS, load_dataset
from src.models import (get_models, export_final, _top1, _subsample,
                        FAMILY, CONFIG, RANDOM_STATE)

RADIUS_M = 50.0
K = 10
INDEX = "artifacts/models.json"


def export_one(name, tr, te):
    import numpy as np
    rng = np.random.default_rng(RANDOM_STATE)
    idx = _subsample(name, np.arange(len(tr)), rng)

    model = get_models()[name]
    t0 = time.time()
    model.fit(tr[FEATURE_COLS].iloc[idx], tr.label.iloc[idx])
    fit_s = time.time() - t0

    t0 = time.time()
    p = model.predict_proba(te[FEATURE_COLS])[:, 1]
    inf_s = time.time() - t0

    meta = {
        "model_name": name,
        "family": FAMILY[name],
        "config": CONFIG[name],
        "radius_m": RADIUS_M,
        "k": K,
        "trained_rows": int(len(idx)),
        "trained_trips": int(tr.trip_id.nunique()),
        "student_test_top1": float(_top1(te, p)),
        "student_test_macro_f1": float(f1_score(te.label, (p >= 0.5).astype(int),
                                                average="macro", zero_division=0)),
        "student_test_log_loss": float(log_loss(te.label, p, labels=[0, 1])),
        "fit_seconds": round(fit_s, 1),
        "inference_seconds": round(inf_s, 2),
    }

    path = f"artifacts/model_{name}.pkl"
    export_final(model, FEATURE_COLS, path, extra=meta)
    size_mb = os.path.getsize(path) / 1e6
    meta["artifact"] = path
    meta["size_mb"] = round(size_mb, 1)
    print(f"  {name:24} top1={meta['student_test_top1']:.4f} "
          f"f1={meta['student_test_macro_f1']:.4f} "
          f"logloss={meta['student_test_log_loss']:.4f} "
          f"({size_mb:.1f} MB, fit {fit_s:.0f}s)", flush=True)
    return meta


def main(names=None):
    tr = load_dataset("results/dataset_train.pkl")
    te = load_dataset("results/dataset_test.pkl")
    print(f"train {len(tr):,} rows / {tr.trip_id.nunique()} trips\n")

    names = names or list(get_models())
    index = {}
    if os.path.exists(INDEX):
        with open(INDEX) as f:
            index = json.load(f)

    for name in names:
        index[name] = export_one(name, tr, te)

    with open(INDEX, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {INDEX} ({len(index)} models)")

    # Point the default at whichever model currently scores highest.
    best = max(index.values(), key=lambda m: m["student_test_top1"])
    shutil.copyfile(best["artifact"], "artifacts/model.pkl")
    print(f"default artifacts/model.pkl -> {best['model_name']} "
          f"(Top-1 {best['student_test_top1']:.4f})")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
