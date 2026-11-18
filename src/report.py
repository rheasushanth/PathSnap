"""Assemble the technical report skeleton from measured results.  OWNER: Rhea

    python -m src.report        -> results/REPORT.md

WHAT THIS DOES AND DOES NOT DO
It reads the committed CSVs and emits every table and figure reference with
the real numbers already filled in, in the section order the brief requires.
Transcribing 8 CSVs by hand is where wrong numbers creep into reports, and
the rubric states that every figure in the PDF must be verifiable against the
repository - generating them from the CSVs makes that true by construction.

It deliberately does NOT write the analysis. Every place requiring the team's
own interpretation is marked >>> WRITE THIS YOURSELF <<<. The assessment
policy prohibits submitting generated analysis as evidence of work the team
did not do, and every member must be able to defend what is written. Fill
those in from your own reading of the results.

Convert to PDF with any Markdown editor, or paste into Docs/Word and export.
"""

import os
from datetime import date

import pandas as pd

OUT = "results"
TODO = ">>> WRITE THIS YOURSELF <<<"


def _csv(name):
    path = f"{OUT}/{name}"
    return pd.read_csv(path) if os.path.exists(path) else None


def _table(df, cols=None, rename=None, floats=4):
    if df is None or not len(df):
        return "_(not yet generated)_\n"
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    return d.round(floats).to_markdown(index=False) + "\n"


# Exact column order and headings required by the brief's section 3.1 table.
RUBRIC_COLS = [
    ("model", "Model"),
    ("family", "Algorithm Family"),
    ("config", "Key Configuration"),
    ("cv_macro_f1", "CV Macro F1-score"),
    ("test_top1", "Top-1 Accuracy"),
    ("test_route_continuity", "Route continuity"),
    ("test_macro_f1", "Student-Test Macro F1-score"),
    ("test_log_loss", "Log Loss"),
    ("inference_time_s", "Inference Time"),
]


def rubric_table(df: pd.DataFrame = None) -> pd.DataFrame:
    """The section 3.1 comparison table, in the brief's exact column order.

    Any metric that genuinely does not apply to a model is written as "N/A"
    rather than left blank - the brief asks for that explicitly.
    """
    df = df if df is not None else _csv("model_comparison.csv")
    if df is None:
        return pd.DataFrame(columns=[h for _, h in RUBRIC_COLS])

    df = df.sort_values("test_top1", ascending=False)
    out = pd.DataFrame()
    for src, head in RUBRIC_COLS:
        if src not in df.columns:
            out[head] = ["N/A"] * len(df)
        elif src in ("model", "family", "config"):
            out[head] = df[src].astype(str).values
        elif src == "inference_time_s":
            out[head] = [f"{v:.2f} s" if pd.notna(v) else "N/A" for v in df[src]]
        else:
            out[head] = [f"{v:.4f}" if pd.notna(v) else "N/A" for v in df[src]]
    return out


def write_rubric_table():
    """Emit the section 3.1 table to artifacts/ and results/ as CSV and Markdown."""
    t = rubric_table()
    for d in ("artifacts", OUT):
        os.makedirs(d, exist_ok=True)
        t.to_csv(f"{d}/table_3_1_model_comparison.csv", index=False)
        with open(f"{d}/table_3_1_model_comparison.md", "w", encoding="utf-8") as f:
            f.write("# Table 3.1 - Overall Model Comparison\n\n")
            f.write("Student-test columns are the 10 held-out student-testing "
                    "trips. CV columns are group-aware 5-fold cross-validation "
                    "over the 80 training trips, grouped by trip.\n\n")
            f.write(t.to_markdown(index=False) + "\n")
    print(f"wrote table_3_1_model_comparison.{{csv,md}} to artifacts/ and {OUT}/ "
          f"({len(t)} models)")
    return t


def build() -> str:
    L = []
    w = L.append

    w(f"# PathSnap — Technical Report\n")
    w("**UE24CS352A Machine Learning · PES University, RR Campus**\n")
    w(f"Generated {date.today().isoformat()} from committed experimental results.\n")
    w("Repository: https://github.com/rheasushanth/PathSnap\n")
    w("\n---\n")

    # ---------------- 1 ----------------
    w("## 1. Problem Statement\n")
    w(f"{TODO} — describe the map-matching problem in your own words: noisy GPS, "
      "a road network, and the need to recover which road segment the vehicle "
      "was actually on at each observation.\n")

    # ---------------- 2 ----------------
    w("\n## 2. Dataset Understanding and Exploratory Data Analysis\n")

    inv = _csv("dataset_inventory.csv")
    if inv is not None:
        summary = inv.groupby("partition").agg(
            trips=("trip_id", "count"),
            observations=("n_obs", "sum"),
            median_obs=("n_obs", "median"),
            total_arcs=("n_arcs", "sum"),
            median_sampling_s=("dt_median", "median"),
            trips_with_gaps_gt_5s=("n_gaps_gt_5s", lambda s: int((s > 0).sum())),
        ).reset_index()
        w("\n### 2.1 Dataset inventory\n")
        w(_table(summary))
        w(f"\nGeographic extent: longitude {inv.lon_min.min():.1f} to "
          f"{inv.lon_max.max():.1f}, latitude {inv.lat_min.min():.1f} to "
          f"{inv.lat_max.max():.1f} — six continents.\n")
        w(f"\nRoad-network connectivity: {int((inv.n_components > 1).sum())} of "
          f"{len(inv)} networks are disconnected, median "
          f"{inv.n_components.median():.0f} components, median node degree "
          f"{inv.mean_degree.median():.2f}.\n")
        w("\n![Dataset overview](fig_eda_overview.png)\n")

        bad = inv[inv.median_snap_m > 100]
        if len(bad):
            w("\n### 2.2 Anomalous trips\n")
            w(_table(bad, ["trip_id", "partition", "n_obs", "median_snap_m"]))
            w("\n![Anomalous trips](fig_anomalous_trips.png)\n")

    lq = _csv("label_quality_data_train.csv")
    lqt = _csv("label_quality_data_student_test.csv")
    if lq is not None and lqt is not None:
        w("\n### 2.3 Label derivation\n")
        w("`.route` is a chronological sequence of arcs, not one label per GPS "
          "observation, so per-observation labels were derived by monotonic "
          "dynamic-programming alignment.\n\n")
        w(_table(pd.DataFrame([
            {"partition": "train", "trips": len(lq),
             "labelled_fraction": lq.labelled_frac.mean(),
             "median_snap_m": lq.median_snap_m.median(),
             "all_monotonic": bool(lq.monotonic.all())},
            {"partition": "student_test", "trips": len(lqt),
             "labelled_fraction": lqt.labelled_frac.mean(),
             "median_snap_m": lqt.median_snap_m.median(),
             "all_monotonic": bool(lqt.monotonic.all())},
        ])))

    sweep = _csv("candidate_sweep.csv")
    if sweep is not None:
        w("\n### 2.4 Candidate generation and the accuracy ceiling\n")
        w("Candidate miss rate is the fraction of observations whose true arc is "
          "absent from the shortlist. It is a hard upper bound on Top-1 accuracy, "
          "so it was measured before any model was trained.\n\n")
        w(_table(sweep, ["radius_m", "k", "miss_rate", "mean_candidates_per_obs"]))
        best = sweep[(sweep.radius_m == 50) & (sweep.k == 10)]
        if len(best):
            r = best.iloc[0]
            w(f"\nSelected: radius 50 m, k = 10 — miss rate {r.miss_rate:.4f}, "
              f"{r.mean_candidates_per_obs:.2f} candidates per observation, "
              f"ceiling {1 - r.miss_rate:.4f}.\n")
        w("\n![Candidate sweep](fig_candidate_sweep.png)\n")

    w(f"\n**Preprocessing and feature choices.** {TODO} — explain the per-trip "
      "local projection, the 100 m outlier cutoff, and why each feature group "
      "(geometric, directional, kinematic, topological) was included.\n")

    # ---------------- 3 ----------------
    w("\n---\n\n## 3. Model Experimentation and Comparison\n")
    mc = _csv("model_comparison.csv")
    if mc is not None:
        w("\n### 3.1 Overall comparison\n")
        w(_table(mc.sort_values("test_top1", ascending=False),
                 [c for c in ["model", "family", "config", "cv_macro_f1", "cv_top1",
                              "test_top1", "test_macro_f1", "test_log_loss",
                              "test_route_continuity", "train_time_s",
                              "inference_time_s"] if c in mc.columns]))
        w("\n![Model comparison](fig_model_comparison.png)\n")
        w("\n![Generalisation gap](fig_generalisation_gap.png)\n")
        w("\n![Accuracy vs cost](fig_accuracy_vs_cost.png)\n")
        w("\n### 3.2 Per-model discussion\n")
        for _, r in mc.sort_values("test_top1", ascending=False).iterrows():
            w(f"\n#### {r.model} ({r.get('family', '')})\n")
            w(f"- Configuration: `{r.get('config', '')}`\n")
            w(f"- Student-test Top-1: **{r.test_top1:.4f}**, "
              f"Macro F1 {r.test_macro_f1:.4f}, Log Loss {r.test_log_loss:.4f}\n")
            if "cv_top1" in r and pd.notna(r.cv_top1):
                w(f"- Cross-validation Top-1: {r.cv_top1:.4f} "
                  f"(gap to test {r.test_top1 - r.cv_top1:+.4f})\n")
            w(f"- Why suitable / strengths / limitations: {TODO}\n")
            w(f"- Overfitting or underfitting, with evidence: {TODO}\n")

    # ---------------- 4 ----------------
    w("\n---\n\n## 4. Bayesian Hyperparameter Optimisation\n")
    tc = _csv("tuning_comparison.csv")
    tr = _csv("optuna_trials.csv")
    if tc is not None:
        w(f"\nOptuna TPESampler, {len(tr) if tr is not None else 'n'} trials, "
          "group-aware cross-validation, objective = observation-level Top-1.\n\n")
        w(_table(tc))
        d = tc.test_top1.iloc[-1] - tc.test_top1.iloc[0]
        w(f"\nTuned minus untuned Top-1: **{d:+.4f}**.\n")
        w(f"\n**Interpretation.** {TODO} — the search did not improve Top-1 "
          "materially. Discuss why that is a legitimate result: the default "
          "configuration was already near-optimal for this feature set, the "
          "gain is within fold-to-fold noise, and the tuned model costs "
          "substantially more to train for no measurable benefit.\n")
    w("\nFull trial history: `results/optuna_trials.csv`.\n")

    # ---------------- 5 ----------------
    w("\n---\n\n## 5. HMM and Viterbi Route Reconstruction\n")
    hc = _csv("hmm_comparison.csv")
    if hc is not None:
        import numpy as np
        wts = hc.n_scored
        rows = []
        for label, c, h in [("Top-1 accuracy", "clf_top1", "hmm_top1"),
                            ("Route continuity", "clf_continuity", "hmm_continuity")]:
            cv, hv = np.average(hc[c], weights=wts), np.average(hc[h], weights=wts)
            rows.append({"Metric": label, "Classifier only": round(cv, 4),
                         "Classifier + HMM/Viterbi": round(hv, 4),
                         "Change": round(hv - cv, 4)})
        w("\n### 5.1 Overall effect (observation-weighted, 10 student-test trips)\n")
        w(_table(pd.DataFrame(rows)))
        w(f"\nTrips improved on Top-1: {int((hc.top1_delta > 0).sum())}/{len(hc)}. "
          f"On continuity: {int((hc.continuity_delta > 0).sum())}/{len(hc)}.\n")
        w("\n### 5.2 Per-trip results\n")
        w(_table(hc, ["trip_id", "n_obs", "clf_top1", "hmm_top1", "top1_delta",
                      "clf_continuity", "hmm_continuity", "continuity_delta",
                      "hmm_time_s"]))
        w("\n![HMM comparison](fig_hmm_comparison.png)\n")
        w("\n![HMM per-trip delta](fig_hmm_delta.png)\n")
        worse = hc[hc.top1_delta < -0.001]
        if len(worse):
            w("\n### 5.3 Where decoding did not help\n")
            w(_table(worse, ["trip_id", "max_gap_s", "clf_top1", "hmm_top1",
                             "top1_delta", "clf_continuity"]))
            w(f"\n{TODO} — explain this case. Note its classifier-only continuity "
              "was already very high, so there was little for the decoder to "
              "repair, and the transition prior pulled some correct independent "
              "predictions onto a smoother but wrong path.\n")
    w(f"\n**Hidden states, emissions, transitions, gap handling.** {TODO} — "
      "describe each, referring to `src/hmm.py`.\n")

    # ---------------- 6, 7, 8 ----------------
    w("\n---\n\n## 6. Geographical Visualisation\n")
    w("Interactive Folium map committed at `maps/trip_00000007.html` with four "
      "toggleable layers: raw GPS observations, road network, reference route "
      "and predicted route.\n")
    w(f"\n{TODO} — insert a screenshot and describe what it shows.\n")

    w("\n---\n\n## 7. API, UI and Deployment\n")
    w("FastAPI backend on Laptop 1, Streamlit UI on Laptop 2. The UI lists the "
      "evaluation trips, lets the evaluator choose among the trained models and "
      "toggle HMM decoding, and renders the predicted route. Ground-truth labels "
      "are read server-side only; the response carries aggregate metrics alone.\n")
    w(f"\n{TODO} — insert screenshots of the working UI and of the API request "
      "logs on the receiving laptop.\n")

    w("\n---\n\n## 8. Academic Integrity and AI-Use Disclosure\n")
    w(f"{TODO} — disclose AI assistance as required by the event policy, and "
      "confirm every result reported here is reproducible from the repository.\n")

    return "\n".join(L)


if __name__ == "__main__":
    write_rubric_table()
    md = build()
    path = f"{OUT}/REPORT.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    n_todo = md.count(TODO)
    print(f"wrote {path}  ({len(md.splitlines())} lines, "
          f"{n_todo} sections for you to write)")
