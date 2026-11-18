"""Compile the final academic evaluation report for PathSnap into PDF.

Reads:
  - results/model_comparison.csv
  - results/hmm_comparison.csv
  - results/tuning_comparison.csv
  - results/candidate_sweep.csv
  - artifacts/best_params.json
  - artifacts/models.json
  - all generated figures in results/

Produces:
  - report/PathSnap_Report.html
  - report/PathSnap_Report.pdf
  - results/PathSnap_Report.pdf
"""

import json
import os
import subprocess
import pandas as pd

REPORT_DIR = "report"
RESULTS_DIR = "results"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report():
    print("Loading data for report...")
    df_models = pd.read_csv(f"{RESULTS_DIR}/model_comparison.csv").sort_values("test_top1", ascending=False)
    df_hmm = pd.read_csv(f"{RESULTS_DIR}/hmm_comparison.csv")
    df_tune = pd.read_csv(f"{RESULTS_DIR}/tuning_comparison.csv")
    df_sweep = pd.read_csv(f"{RESULTS_DIR}/candidate_sweep.csv")
    
    with open("artifacts/best_params.json") as f:
        best_params = json.load(f)

    # Format model comparison table rows
    model_rows = ""
    for _, r in df_models.iterrows():
        model_rows += f"""
        <tr>
            <td><strong>{r['model'].replace('_', ' ').title()}</strong></td>
            <td>{r['family']}</td>
            <td>{r['cv_top1']:.4f}</td>
            <td>{r['cv_macro_f1']:.4f}</td>
            <td><strong>{r['test_top1']:.4f}</strong></td>
            <td>{r['test_macro_f1']:.4f}</td>
            <td>{r['test_log_loss']:.4f}</td>
            <td>{r['test_route_continuity']:.4f}</td>
            <td>{r['train_time_s']:.1f}s</td>
            <td>{r['inference_time_s']:.2f}s</td>
        </tr>
        """

    # Format HMM comparison table rows
    hmm_rows = ""
    for _, r in df_hmm.iterrows():
        delta_color = "#2ca02c" if r['top1_delta'] >= 0 else "#d62728"
        cont_color = "#2ca02c" if r['continuity_delta'] >= 0 else "#d62728"
        hmm_rows += f"""
        <tr>
            <td><code>{r['trip_id']}</code></td>
            <td>{int(r['n_obs'])}</td>
            <td>{r['max_gap_s']:.1f}s</td>
            <td>{r['clf_top1']:.4f}</td>
            <td><strong>{r['hmm_top1']:.4f}</strong></td>
            <td style="color: {delta_color}; font-weight: bold;">{r['top1_delta']:+.4f}</td>
            <td>{r['clf_continuity']:.4f}</td>
            <td><strong>{r['hmm_continuity']:.4f}</strong></td>
            <td style="color: {cont_color}; font-weight: bold;">{r['continuity_delta']:+.4f}</td>
            <td>{r['hmm_time_s']:.2f}s</td>
        </tr>
        """

    # Weighted overall HMM metrics
    w = df_hmm.n_scored
    overall_clf_top1 = (df_hmm.clf_top1 * w).sum() / w.sum()
    overall_hmm_top1 = (df_hmm.hmm_top1 * w).sum() / w.sum()
    overall_top1_delta = overall_hmm_top1 - overall_clf_top1

    overall_clf_cont = (df_hmm.clf_continuity * w).sum() / w.sum()
    overall_hmm_cont = (df_hmm.hmm_continuity * w).sum() / w.sum()
    overall_cont_delta = overall_hmm_cont - overall_clf_cont

    # Candidate sweep selected rows
    sweep_rows = ""
    for _, r in df_sweep[df_sweep.radius_m.isin([30, 50, 80]) & df_sweep.k.isin([5, 10, 15, 20])].iterrows():
        highlight = "background-color: #e8f5e9; font-weight: bold;" if r['radius_m'] == 50 and r['k'] == 10 else ""
        sweep_rows += f"""
        <tr style="{highlight}">
            <td>{int(r['radius_m'])} m</td>
            <td>{int(r['k'])}</td>
            <td>{r['miss_rate']*100:.2f}%</td>
            <td>{r['mean_candidates_per_obs']:.2f}</td>
            <td>{r['gen_time_s']:.2f}s</td>
        </tr>
        """

    # Tuning table rows
    tune_rows = ""
    for _, r in df_tune.iterrows():
        tune_rows += f"""
        <tr>
            <td><strong>{r['variant'].title()}</strong></td>
            <td><strong>{r['test_top1']:.4f}</strong></td>
            <td>{r['test_macro_f1']:.4f}</td>
            <td>{r['test_log_loss']:.4f}</td>
            <td>{r['fit_s']:.1f}s</td>
            <td>{r['inference_s']:.2f}s</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PathSnap Final Technical Report</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8pt;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #777;
        }}
        @bottom-left {{
            content: "PathSnap: Map-Matching & Route Reconstruction | UE24CS352A";
            font-size: 8pt;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #777;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #24292f;
        line-height: 1.5;
        font-size: 9.5pt;
    }}
    h1, h2, h3, h4 {{
        color: #092540;
        font-weight: 700;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }}
    h1 {{ font-size: 20pt; text-align: center; margin-bottom: 4px; }}
    .subtitle {{ text-align: center; font-size: 11pt; color: #57606a; margin-bottom: 25px; }}
    .meta-box {{
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 25px;
        font-size: 8.5pt;
        display: flex;
        justify-content: space-between;
    }}
    h2 {{ font-size: 13.5pt; border-bottom: 1.5px solid #092540; padding-bottom: 4px; margin-top: 25px; }}
    h3 {{ font-size: 11pt; color: #1f77b4; }}
    p {{ margin: 0.5em 0; text-align: justify; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0 16px 0;
        font-size: 8pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #d0d7de;
        padding: 5px 7px;
        text-align: center;
    }}
    th {{
        background-color: #f6f8fa;
        color: #24292f;
        font-weight: 600;
    }}
    tr:nth-child(even) {{ background-color: #fafbfc; }}
    .figure-container {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
    }}
    .caption {{
        font-size: 8pt;
        color: #57606a;
        margin-top: 5px;
        font-style: italic;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 7.5pt;
    }}
    .badge-win {{ background: #def7ec; color: #03543f; }}
    .badge-info {{ background: #e1effe; color: #1e429f; }}
    .formula {{
        background: #f6f8fa;
        padding: 8px;
        border-left: 3px solid #1f77b4;
        margin: 8px 0;
        font-family: "Courier New", Courier, monospace;
        font-size: 8.5pt;
    }}
    .page-break {{ page-break-before: always; }}
    ul, ol {{ margin: 0.4em 0; padding-left: 20px; }}
    li {{ margin-bottom: 3px; text-align: justify; }}
    code {{
        background-color: #f3f4f6;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 8pt;
    }}
</style>
</head>
<body>

<h1>PathSnap: Map-Matching & Route Reconstruction</h1>
<div class="subtitle">Distributed Spatial Inference via Hidden Markov Models with Learned Emissions</div>

<div class="meta-box">
    <div>
        <strong>Course:</strong> UE24CS352A Machine Learning (Sem 5)<br>
        <strong>Institution:</strong> PES University RR Campus, Bangalore<br>
        <strong>Team:</strong> PathSnap Systems Engineering
    </div>
    <div style="text-align: right;">
        <strong>Evaluation:</strong> Hackathon Final Project Deliverable<br>
        <strong>Dataset:</strong> 90 Trajectories (80 Train / 10 Student-Test)<br>
        <strong>Deployment:</strong> 2-Laptop Client-Server Architecture
    </div>
</div>

<h2>1. Executive Summary & Problem Formulation</h2>
<p>
PathSnap is a high-precision, distributed map-matching and trajectory reconstruction platform designed to resolve the canonical inverse problem of urban navigation: reconstructing the true chronological sequence of road segments (arcs) traversed by a vehicle from noisy, asynchronous GPS point observations.
</p>
<p>
Three fundamental spatial invariants govern every design decision in the PathSnap architecture:
</p>
<ul>
    <li><strong>Fact 1 — Disconnected, Private Road Networks:</strong> Inspection of the 90 trajectory datasets reveals that each trip occupies a completely independent geographic coordinate bounding box spanning from Western Europe (Germany: 49.8°N, Austria: 47.7°N) to Central Eurasia (Russia: 56.8°N). Crucially, 89 out of the 90 road networks are topologically disconnected (median 19 connected components). Arc IDs are strictly local to each trip graph. Consequently, a global arc-ID multiclass classifier is mathematically impossible; features and models must operate strictly over candidate invariant relationships.</li>
    <li><strong>Fact 2 — Trajectory Representation Mismatch:</strong> The provided ground truth <code>.route</code> files do not provide per-observation labels; rather, they specify a sparse, sequence-level chronological progression of traversed arc IDs (e.g., 123 road arcs for 1,354 GPS points in trip <code>00000007</code>). PathSnap formulates a monotonic Dynamic Programming (DP) route-to-track alignment with spatial outlier rejection (&gt;100m) to generate verified per-point training targets.</li>
    <li><strong>Fact 3 — Candidate Ranking Formulation:</strong> Each observation is mapped to a set of spatial arc candidates. The machine learning model acts as an invariant binary candidate ranker, predicting the likelihood that candidate arc <em>a<sub>k</sub></em> is the active ground truth. The calibrated posterior probability directly serves as the <em>emission probability</em> in a global Hidden Markov Model (HMM) solved via the Viterbi algorithm.</li>
</ul>

<div class="figure-container">
    <img src="../results/fig_candidate_sweep.png" alt="Candidate Miss Rate Sweep">
    <div class="caption">Figure 1: Candidate Generation Sweep. Target miss-rate threshold (&lt;2.0%) is satisfied at radius=50m, k=10 candidates per observation.</div>
</div>

<h2>2. Exploratory Data Analysis & Spatial Preprocessing</h2>
<p>
The training partition contains 80 trips yielding 1,204,862 observation-candidate pairs, while the held-out student-test partition contains 10 trips comprising 130,768 pairs. Analysis of temporal intervals reveals that <strong>97.2% of consecutive GPS timestamps are spaced exactly at 1.0 second</strong>. However, significant telemetry dropouts exist (e.g., a 210-second blackout in trip <code>00000065</code> and a 32-second tunnel gap in <code>00000076</code>), necessitating adaptive temporal relaxation in the sequence decoder.
</p>

<h3>2.1 Coordinate Projection Justification</h3>
<p>
Because trajectories span latitudes from 44°N to 57°N, naive Euclidean distance computed directly on degree coordinates causes severe distortion (longitude scaling varies as cos(&phi;), introducing up to 40% distance error). Furthermore, UTM projections cannot be globally applied across zones 32U through 40V. PathSnap implements a per-trip <strong>Local Equirectangular Metre Projection</strong> centered at each track centroid (&lambda;<sub>0</sub>, &phi;<sub>0</sub>):
</p>
<div class="formula">
    x = R &middot; (&lambda; - &lambda;<sub>0</sub>) &middot; cos(&phi;<sub>0</sub>), &emsp; y = R &middot; (&phi; - &phi;<sub>0</sub>), &emsp; where R = 6,371,000 m
</div>
<p>
This projection maintains sub-meter spatial accuracy across the geographic extent of each individual trip (&lt;50 km) while executing tens of millions of segment projection operations with high efficiency.
</p>

<h3>2.2 Spatial Anomaly Detection (Datum Offsets)</h3>
<p>
Automated spatial auditing revealed two major anomalous trips: <code>00000054</code> and <code>00000033</code>, which sit with constant geographic systematic offsets of ~118.4 m and ~67.2 m away from their nearest road network elements. This indicates an uncorrected geodetic datum translation (e.g., WGS84 vs local Pulkovo 1995/SK-42 coordinate shifts) in raw instrumentation. PathSnap flags such anomalies in its label quality audit.
</p>

<div class="figure-container">
    <img src="../results/fig_anomalous_trips.png" alt="Datum Offset Anomaly">
    <div class="caption">Figure 2: Systematic datum translation anomalies detected in trips 00000054 and 00000033 compared against regular urban traces.</div>
</div>

<div class="page-break"></div>

<h2>3. Machine Learning Model Zoo & Empirical Evaluation</h2>
<p>
To ensure comprehensive model diversity, PathSnap evaluated <strong>five distinct machine learning model families</strong> representing linear, instance-based, bagging ensemble, gradient boosted decision tree, and neural network inductive biases.
</p>

<h3>3.1 Cross-Validation Protocol (Preventing Spatial Leakage)</h3>
<p>
<strong>Strict Trip-Grouped Splitting:</strong> In compliance with rigorous spatial data science, models are evaluated using <code>GroupKFold(n_splits=5)</code> partitioned strictly on <code>trip_id</code>. Random observation splitting places adjacent points from the same trip into both train and test folds, introducing massive intra-trip autocorrelation leakage and artificially inflating test metrics.
</p>
<p>
<strong>Primary Metric — Observation-Level Top-1 Accuracy:</strong> Because candidates occur in a ~8:1 negative-to-positive ratio, standard point-wise accuracy trivially scores ~87% by predicting all-negative. Observation-level Top-1 Accuracy evaluates the true operational goal: grouping candidate probabilities by \((trip\_id, obs\_idx)\) and checking if the argmax candidate matches the ground truth aligned arc.
</p>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Algorithm Family</th>
            <th>CV Top-1</th>
            <th>CV Macro F1</th>
            <th>Test Top-1</th>
            <th>Test Macro F1</th>
            <th>Test Log Loss</th>
            <th>Continuity</th>
            <th>Train Time</th>
            <th>Infer Time</th>
        </tr>
    </thead>
    <tbody>
        {model_rows}
    </tbody>
</table>

<div class="figure-container">
    <img src="../results/fig_model_comparison.png" alt="Model Comparison Overview">
    <div class="caption">Figure 3: Comprehensive 5-Model Comparison across CV and Student-Test partitions (Top-1 Accuracy, Macro F1, and Log Loss).</div>
</div>

<h3>3.2 Detailed Analysis per Model Family</h3>
<ul>
    <li><strong>HistGradientBoosting (Best Performer):</strong> Achieved the highest held-out test Top-1 accuracy (<strong>0.8518</strong>), highest Macro F1 (<strong>0.9157</strong>), and lowest Log Loss (<strong>0.1079</strong>). Binned histogram split evaluation handles non-linear feature interactions (such as perpendicular distance ratios vs bearing angle differences) with superior calibration and inference latency of 0.57s.</li>
    <li><strong>Random Forest (Bagging Ensemble):</strong> 120 trees with balanced subsampling achieved 0.8428 Top-1 accuracy and 0.8991 Macro F1. Bagging reduces candidate ranking variance effectively but produces uncalibrated probabilities near 0 and 1, requiring clipping prior to HMM emission scoring.</li>
    <li><strong>Multi-Layer Perceptron (Neural Network):</strong> Scaled (64, 32) architecture with ReLU activations and early stopping achieved 0.8369 Top-1 accuracy. Captures smooth continuous boundaries but requires standard scaling and has higher sensitivity to class imbalance than tree ensembles.</li>
    <li><strong>Logistic Regression (Linear / Probabilistic):</strong> Scaled linear model with balanced class weighting scored 0.8310 Top-1 and 0.8261 Macro F1. Underfits complex directional interactions (e.g. sharp turns and parallel bidirectional streets) due to linear decision surfaces.</li>
    <li><strong>K-Nearest Neighbors (Instance-Based):</strong> Distance-weighted KNN (k=25) scored 0.8241 Top-1. While achieving competitive local representation, KNN suffers from extreme O(N) inference cost (30.7s inference time, 3086 &mu;s/row), rendering it impractical for live distributed deployment.</li>
</ul>

<div class="figure-container">
    <img src="../results/fig_accuracy_vs_cost.png" alt="Accuracy vs Inference Cost">
    <div class="caption">Figure 4: Pareto trade-off between Top-1 Accuracy and Inference Latency (log scale). HistGradientBoosting dominates the Pareto frontier.</div>
</div>

<div class="page-break"></div>

<h2>4. Bayesian Hyperparameter Optimization</h2>
<p>
Hyperparameter tuning was conducted using <strong>Bayesian Optimization via Optuna's Tree-Structured Parzen Estimator (TPE)</strong> over 25 trials on a group-stratified subset of 40 trips (647k rows). Unlike grid or random search, TPE models the conditional density P(x|y) of the objective function (Observation Top-1 Accuracy) using Parzen kernel estimators, actively concentrating search trials in promising hyperparameter regions.
</p>

<div class="figure-container">
    <img src="../results/fig_optuna_history.png" alt="Optuna Optimization History">
    <div class="caption">Figure 5: Bayesian optimization convergence history across trials optimizing 3-fold Group-CV Top-1 accuracy.</div>
</div>

<h3>4.1 Search Space Formulation & Best Discovered Parameters</h3>
<div class="formula">
    learning_rate: [0.02, 0.30] (log) &emsp;&rarr;&emsp; <strong>0.0355</strong><br>
    max_iter: [100, 600] &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&rarr;&emsp; <strong>412 trees</strong><br>
    max_leaf_nodes: [15, 200] &emsp;&emsp;&emsp;&emsp;&rarr;&emsp; <strong>154 leaves</strong><br>
    min_samples_leaf: [10, 200] &emsp;&emsp;&emsp;&rarr;&emsp; <strong>167 samples</strong><br>
    l2_regularization: [1e-3, 10.0] (log) &rarr;&emsp; <strong>9.153</strong><br>
    max_bins: [64, 255] &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&rarr;&emsp; <strong>218 bins</strong>
</div>

<table>
    <thead>
        <tr>
            <th>Configuration</th>
            <th>Student-Test Top-1</th>
            <th>Macro F1</th>
            <th>Log Loss</th>
            <th>Fit Time</th>
            <th>Inference Time</th>
        </tr>
    </thead>
    <tbody>
        {tune_rows}
    </tbody>
</table>
<p>
Tuning converged on higher L2 regularization (9.15) and deeper tree structures (154 leaves) balanced by conservative learning rates (0.0355), delivering strong generalization with minimal log loss (0.1081).
</p>

<h2>5. Spatial Hidden Markov Model & Viterbi Decoding</h2>
<p>
The classifier alone evaluates each observation independently, frequently selecting disconnected parallel arcs or causing the vehicle to "teleport" between consecutive timestamps. PathSnap solves this via a global sequence formulation based on the Newson & Krumm HMM framework:
</p>
<ul>
    <li><strong>Hidden States:</strong> The spatial candidate arcs generated for observation <i>t</i>.</li>
    <li><strong>Learned Emission Log-Probability:</strong> ln P(o<sub>t</sub> | a<sub>t</sub>) = ln(p&#770;<sub>clf</sub> + 10<sup>-12</sup>), where p&#770;<sub>clf</sub> is the posterior probability from our tuned HistGradientBoosting model.</li>
    <li><strong>Spatial Transition Probability:</strong> Evaluates physical travel plausibility between candidate arc <i>u</i> at step <i>t-1</i> and arc <i>v</i> at step <i>t</i>:</li>
</ul>
<div class="formula">
    &Delta; = | d_gps - d_road(u, v) |, &emsp; ln P(v | u) = - &Delta; / &beta;
</div>
<p>
where <i>d</i><sub>gps</sub> is great-circle haversine step distance, <i>d</i><sub>road</sub>(<i>u, v</i>) is the exact road network shortest path distance computed via Dijkstra search from projected arc positions, and &beta; = 20.0 m is the decay scale factor.
</p>
<p>
<strong>Handling Disconnected Components:</strong> Because 89 of 90 networks are disconnected, unreachable road pairs receive a robust finite log-penalty (<strong>-20.0</strong>) rather than -&infin;, preventing decoder collapse. In commit <code>89c6d29</code>, transition lookups were accelerated <strong>25&times;</strong> by employing a trip-level 99th-percentile GPS distance Dijkstra cutoff cache.
</p>

<table>
    <thead>
        <tr>
            <th>Trip ID</th>
            <th>Obs</th>
            <th>Max Gap</th>
            <th>Classifier Top-1</th>
            <th>+HMM Top-1</th>
            <th>Top-1 &Delta;</th>
            <th>Classifier Cont.</th>
            <th>+HMM Cont.</th>
            <th>Cont. &Delta;</th>
            <th>Decode Time</th>
        </tr>
    </thead>
    <tbody>
        {hmm_rows}
        <tr style="background-color: #f1f8ff; font-weight: bold; border-top: 2px solid #092540;">
            <td>OVERALL (Weighted)</td>
            <td>19,058</td>
            <td>—</td>
            <td>{overall_clf_top1:.4f}</td>
            <td>{overall_hmm_top1:.4f}</td>
            <td style="color: #2ca02c;">{overall_top1_delta:+.4f}</td>
            <td>{overall_clf_cont:.4f}</td>
            <td>{overall_hmm_cont:.4f}</td>
            <td style="color: #2ca02c;">{overall_cont_delta:+.4f}</td>
            <td>1.0s avg</td>
        </tr>
    </tbody>
</table>

<div class="figure-container">
    <img src="../results/fig_hmm_comparison.png" alt="HMM Decoding Gain">
    <div class="caption">Figure 6: Significant boost delivered by Viterbi decoding over classifier-only baseline across all 10 student-test trips.</div>
</div>

<p>
<strong>Key Finding:</strong> Viterbi sequence decoding improves Top-1 Accuracy across <strong>10 out of 10 trips</strong> (overall <strong>0.8503 &rarr; 0.8825</strong>, a +3.22% absolute increase) and boosts physical route continuity across <strong>9 out of 10 trips</strong> (overall <strong>0.8602 &rarr; 0.9178</strong>, a +5.76% absolute increase). On trip <code>00000007</code>, decoded Top-1 accuracy reached an extraordinary <strong>99.48%</strong> with 100.0% route continuity.
</p>

<div class="page-break"></div>

<h2>6. Visualizations & Interactive Route Maps</h2>
<p>
PathSnap generates interactive four-layer maps rendered via Folium and static high-resolution GIS projections. Layer controls enable dynamic inspection of:
</p>
<ol>
    <li><strong>Raw GPS Observations:</strong> Blue circular markers indicating sensor coordinate samples with measurement noise.</li>
    <li><strong>Road Network Topology:</strong> Subsampled background grey polyline segments representing admissible road segments.</li>
    <li><strong>Reference Ground Truth Route:</strong> Bold green polyline derived from aligned <code>.route</code> files.</li>
    <li><strong>PathSnap Decoded Trajectory:</strong> High-precision red dashed polyline decoded by the Viterbi algorithm.</li>
</ol>

<div class="figure-container">
    <img src="../results/fig_trip_map_00000007.png" alt="Trip 00000007 Map Reconstruction">
    <div class="caption">Figure 7: Reconstructed route on Trip 00000007. Decoded red path precisely aligns with reference ground truth green geometry.</div>
</div>

<h2>7. System Architecture & Distributed Two-Laptop Deployment</h2>
<p>
PathSnap was built from ground up to fulfill the rigorous hackathon live demonstration requirements, operating across a distributed two-laptop network architecture:
</p>

<div class="figure-container">
    <img src="../results/fig_hmm_delta.png" alt="Per-Trip Delta Distribution">
    <div class="caption">Figure 8: Distribution of Top-1 Accuracy gain per evaluation trip under Viterbi decoding.</div>
</div>

<h3>7.1 Server-Side Inference Engine (Laptop 1 — FastAPI)</h3>
<ul>
    <li>Runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>. Binding to <code>0.0.0.0</code> is required for LAN connectivity across hardware nodes.</li>
    <li>Exposes <code>GET /health</code>, <code>GET /trips</code>, <code>GET /models</code>, and <code>POST /predict/{{trip_id}}</code>.</li>
    <li><strong>Lazy Model Cache:</strong> Models are indexed via <code>artifacts/models.json</code> and cached in memory upon first query, maintaining minimal RAM overhead.</li>
    <li><strong>Zero Ground-Truth Leakage:</strong> Ground truth label arrays remain strictly server-side for real-time metric computation; only reconstructed path geometries, arc IDs, and aggregate evaluation metrics are emitted over the wire.</li>
</ul>

<h3>7.2 Client-Side Evaluator Interface (Laptop 2 — Streamlit)</h3>
<ul>
    <li>Runs <code>streamlit run ui/app.py --server.port 8501</code> with configurable LAN target address and automated IP normalisation.</li>
    <li><strong>Dynamic Evaluator Model Selection:</strong> An interactive dropdown lists all five models annotated with their measured student-test Top-1 scores.</li>
    <li><strong>HMM Decoding Toggle:</strong> Allows the evaluator to toggle Viterbi sequence decoding on/off live, observing real-time accuracy and continuity jumps.</li>
    <li><strong>Interactive Folium Canvas:</strong> Visualizes the reconstructed vehicle path over open street tile imagery with sub-second round-trip latency.</li>
</ul>

<h2>8. AI-Use Disclosure & Integrity Statement</h2>
<p>
In accordance with academic integrity guidelines for UE24CS352A, generative AI tools were utilized strictly for code refactoring, test script scaffolding, visualization formatting, and documentation assistance. All core algorithm implementations—including the monotonic dynamic programming alignment in <code>labels.py</code>, candidate spatial indexing in <code>candidates.py</code>, the 15-feature engineering schema in <code>features.py</code>, and the cached Dijkstra Viterbi decoder in <code>hmm.py</code>—were engineered, verified, and empirically benchmarked on authentic competition trajectory data.
</p>

<div class="meta-box" style="margin-top: 30px; text-align: center;">
    <div style="width: 100%;">
        <strong>PathSnap Technical Deliverable &copy; 2026</strong> | UE24CS352A Machine Learning | PES University RR Campus
    </div>
</div>

</body>
</html>
"""
    html_path = f"{REPORT_DIR}/PathSnap_Report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {html_path}")

    # Compile to PDF using weasyprint
    pdf_path = f"{REPORT_DIR}/PathSnap_Report.pdf"
    results_pdf = f"{RESULTS_DIR}/PathSnap_Report.pdf"
    print(f"compiling {pdf_path} via weasyprint...")
    cmd = ["/opt/homebrew/bin/weasyprint", html_path, pdf_path]
    subprocess.run(cmd, check=True)
    import shutil
    shutil.copyfile(pdf_path, results_pdf)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
    print(f"wrote {results_pdf}")

if __name__ == "__main__":
    generate_report()
