"""Training vs cross-validation vs test accuracy.  OWNER: Rhea

    python -m src.train_vs_test

Answers report section 3.2 question 9 - "did you observe overfitting or
underfitting, and what evidence supports your conclusion?" - with measured
numbers rather than an assertion.

THREE NUMBERS, AND WHY ALL THREE ARE NEEDED
  train  scored on the same rows the model was fitted on. On its own this
         says nothing about generalisation; a model that memorises scores
         1.0 here.
  cv     group-aware 5-fold cross-validation over the 80 training trips,
         grouped by trip. Held out, but from the same distribution.
  test   the 10 student-testing trips, never seen in any form.

  train - cv    measures how much the model fits noise within the training
                trips. A large gap is overfitting.
  cv - test     measures whether the training trips are representative. A
                large gap means the test trips differ, not that the model
                overfits.

A model with train ~= cv ~= test and all three low is underfitting - it is
consistent, just not capable enough.

Scores the exported artifacts, so nothing is refitted and the numbers refer
to exactly the models the API serves.
"""

import json
import os

import pandas as pd
from sklearn.metrics import f1_score

from src.features import FEATURE_COLS, load_dataset
from src.models import _top1, load_final

OUT = "results"
INDEX = "artifacts/models.json"


def score(model, df, feature_cols):
    p = model.predict_proba(df[feature_cols])[:, 1]
    return {
        "top1": _top1(df, p),
        "macro_f1": f1_score(df.label, (p >= 0.5).astype(int),
                             average="macro", zero_division=0),
    }


def main():
    tr = load_dataset("results/dataset_train.pkl")
    te = load_dataset("results/dataset_test.pkl")
    with open(INDEX) as f:
        index = json.load(f)

    cv = pd.read_csv(f"{OUT}/model_comparison.csv").set_index("model")

    rows = []
    for name, entry in index.items():
        if not os.path.exists(entry["artifact"]):
            continue
        bundle = load_final(entry["artifact"])
        model, feats = bundle["model"], bundle["features"]
        print(f"  scoring {name}...", flush=True)

        s_tr = score(model, tr, feats)
        s_te = score(model, te, feats)
        cv_top1 = float(cv.cv_top1.get(name, float("nan")))
        cv_f1 = float(cv.cv_macro_f1.get(name, float("nan")))

        rows.append({
            "model": name,
            "family": entry.get("family"),
            "train_top1": s_tr["top1"],
            "cv_top1": cv_top1,
            "test_top1": s_te["top1"],
            "train_macro_f1": s_tr["macro_f1"],
            "cv_macro_f1": cv_f1,
            "test_macro_f1": s_te["macro_f1"],
            "train_minus_cv": s_tr["top1"] - cv_top1,
            "cv_minus_test": cv_top1 - s_te["top1"],
        })

    df = pd.DataFrame(rows).sort_values("test_top1", ascending=False)
    df.to_csv(f"{OUT}/train_vs_test.csv", index=False)
    print(f"\nwrote {OUT}/train_vs_test.csv")

    show = ["model", "train_top1", "cv_top1", "test_top1",
            "train_minus_cv", "cv_minus_test"]
    print("\n" + df[show].round(4).to_string(index=False))

    _plot(df)
    return df


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = df.sort_values("test_top1", ascending=False)
    x = np.arange(len(d))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, d.train_top1, w, label="Train (fitted rows)", color="#4C72B0")
    ax.bar(x, d.cv_top1, w, label="Cross-validation (grouped by trip)", color="#DD8452")
    ax.bar(x + w, d.test_top1, w, label="Student-test (held out)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in d.model], fontsize=8)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Training vs cross-validation vs held-out accuracy\n"
                 "(the train-to-CV gap is the overfitting evidence)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(d.itertuples()):
        ax.text(i - w, r.train_top1, f"{r.train_top1:.3f}", ha="center",
                va="bottom", fontsize=6)
        ax.text(i + w, r.test_top1, f"{r.test_top1:.3f}", ha="center",
                va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_train_vs_test.png", dpi=150)
    print(f"wrote {OUT}/fig_train_vs_test.png")


if __name__ == "__main__":
    main()
