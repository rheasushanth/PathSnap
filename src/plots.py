"""Model-comparison plots for the report.  OWNER: Rhea -> hand to P3

Every figure is generated from results/model_comparison.csv, which is itself
written by an actual training run. Nothing here is hand-drawn.

The rubric is explicit that training-curve plots must only appear for models
that genuinely produce epoch-wise histories. Of our five, only the MLP does
(loss_curve_), so only the MLP gets one.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

OUT = "results"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _bar(ax, labels, values, title, ylabel, errs=None, fmt="{:.3f}"):
    bars = ax.bar(range(len(labels)), values, yerr=errs, capsize=4,
                  color=PALETTE[:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=7)
    return ax


def plot_model_comparison(csv=f"{OUT}/model_comparison.csv"):
    df = pd.read_csv(csv).sort_values("test_top1", ascending=False)
    names = [m.replace("_", " ") for m in df.model]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _bar(axes[0, 0], names, df.cv_macro_f1, "Cross-validation Macro F1 (80 training trips)",
         "Macro F1", errs=df.get("cv_macro_f1_std"))
    _bar(axes[0, 1], names, df.test_top1, "Student-test Top-1 Accuracy (10 held-out trips)",
         "Top-1 accuracy")
    _bar(axes[1, 0], names, df.test_macro_f1, "Student-test Macro F1", "Macro F1")
    _bar(axes[1, 1], names, df.test_log_loss, "Student-test Log Loss (lower is better)",
         "Log loss", fmt="{:.4f}")
    fig.suptitle("PathSnap model comparison", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_model_comparison.png", dpi=150)
    print(f"wrote {OUT}/fig_model_comparison.png")

    # Accuracy against cost - justifies the selection when scores are close.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.inference_time_s, df.test_top1, s=90, c=PALETTE[:len(df)], zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.model.replace("_", " "),
                    (r.inference_time_s, r.test_top1),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time, full student-test set (s, log scale)")
    ax.set_ylabel("Student-test Top-1 accuracy")
    ax.set_title("Accuracy vs inference cost")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_accuracy_vs_cost.png", dpi=150)
    print(f"wrote {OUT}/fig_accuracy_vs_cost.png")

    # CV generalisation gap - evidence for the over/underfitting question.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(df))
    ax.plot(x, df.cv_top1, "o-", label="CV Top-1 (training trips)", color=PALETTE[0])
    ax.plot(x, df.test_top1, "s-", label="Student-test Top-1 (held out)", color=PALETTE[3])
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Generalisation gap: cross-validation vs held-out trips")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_generalisation_gap.png", dpi=150)
    print(f"wrote {OUT}/fig_generalisation_gap.png")
    return df


def plot_candidate_sweep(csv=f"{OUT}/candidate_sweep.csv"):
    """Miss rate against candidate-set size - justifies the radius/k choice."""
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (radius, g) in enumerate(df.groupby("radius_m")):
        g = g.sort_values("k")
        ax.plot(g.k, g.miss_rate * 100, "o-", label=f"radius {radius} m",
                color=PALETTE[i % len(PALETTE)])
    ax.axhline(2.0, ls="--", c="grey", lw=1)
    ax.text(df.k.max(), 2.05, "2% target", ha="right", fontsize=8, color="grey")
    ax.set_xlabel("k (candidates per observation)")
    ax.set_ylabel("Candidate miss rate (%)")
    ax.set_title("Candidate miss rate - a hard ceiling on achievable Top-1 accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_candidate_sweep.png", dpi=150)
    print(f"wrote {OUT}/fig_candidate_sweep.png")


def plot_mlp_training_curve(model, path=f"{OUT}/fig_mlp_loss_curve.png"):
    """ONLY the MLP has epoch-wise history. Do not fake curves for the others."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "loss_curve_"):
        print("model has no loss_curve_ - skipping (correct behaviour)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(clf.loss_curve_, label="training loss", color=PALETTE[0])
    if hasattr(clf, "validation_scores_") and clf.validation_scores_ is not None:
        ax2 = ax.twinx()
        ax2.plot(clf.validation_scores_, label="validation score",
                 color=PALETTE[3], ls="--")
        ax2.set_ylabel("validation score")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("MLP training curve (the only model with epoch-wise history)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    import os
    df = plot_model_comparison()
    if os.path.exists(f"{OUT}/candidate_sweep.csv"):
        plot_candidate_sweep()
    print("\n" + df[["model", "cv_top1", "test_top1", "test_macro_f1"]].to_string(index=False))
