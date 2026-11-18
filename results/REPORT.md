# PathSnap — Technical Report

**UE24CS352A Machine Learning · PES University, RR Campus**

Generated 2026-11-18 from committed experimental results.

Repository: https://github.com/rheasushanth/PathSnap


---

## 1. Problem Statement

>>> WRITE THIS YOURSELF <<< — describe the map-matching problem in your own words: noisy GPS, a road network, and the need to recover which road segment the vehicle was actually on at each observation.


## 2. Dataset Understanding and Exploratory Data Analysis


### 2.1 Dataset inventory

| partition    |   trips |   observations |   median_obs |   total_arcs |   median_sampling_s |   trips_with_gaps_gt_5s |
|:-------------|--------:|---------------:|-------------:|-------------:|--------------------:|------------------------:|
| student_test |      10 |          19160 |         1420 |       185041 |                   1 |                       2 |
| train        |      80 |         208417 |         1529 |      3876904 |                   1 |                      45 |


Geographic extent: longitude -100.4 to 137.2, latitude -34.6 to 61.7 — six continents.


Road-network connectivity: 89 of 90 networks are disconnected, median 19 components, median node degree 2.15.


![Dataset overview](fig_eda_overview.png)


### 2.2 Anomalous trips

|   trip_id | partition   |   n_obs |   median_snap_m |
|----------:|:------------|--------:|----------------:|
|        33 | train       |    5844 |         122.383 |
|        54 | train       |   19313 |         140.008 |


![Anomalous trips](fig_anomalous_trips.png)


### 2.3 Label derivation

`.route` is a chronological sequence of arcs, not one label per GPS observation, so per-observation labels were derived by monotonic dynamic-programming alignment.


| partition    |   trips |   labelled_fraction |   median_snap_m | all_monotonic   |
|:-------------|--------:|--------------------:|----------------:|:----------------|
| train        |      80 |              0.9803 |          4.6293 | True            |
| student_test |      10 |              0.9996 |          4.7446 | True            |


### 2.4 Candidate generation and the accuracy ceiling

Candidate miss rate is the fraction of observations whose true arc is absent from the shortlist. It is a hard upper bound on Top-1 accuracy, so it was measured before any model was trained.


|   radius_m |   k |   miss_rate |   mean_candidates_per_obs |
|-----------:|----:|------------:|--------------------------:|
|         30 |   5 |      0.0348 |                    3.6555 |
|         30 |  10 |      0.0207 |                    5.1118 |
|         30 |  15 |      0.0201 |                    5.7024 |
|         30 |  20 |      0.0199 |                    5.9156 |
|         50 |   5 |      0.0311 |                    4.1834 |
|         50 |  10 |      0.0167 |                    6.8331 |
|         50 |  15 |      0.0158 |                    8.5239 |
|         50 |  20 |      0.0155 |                    9.4952 |
|         80 |   5 |      0.0308 |                    4.4662 |
|         80 |  10 |      0.0164 |                    7.9205 |
|         80 |  15 |      0.0155 |                   10.8178 |
|         80 |  20 |      0.0153 |                   13.1887 |
|        120 |   5 |      0.0308 |                    4.6168 |
|        120 |  10 |      0.0162 |                    8.472  |
|        120 |  15 |      0.0154 |                   11.9357 |
|        120 |  20 |      0.0151 |                   15.1015 |


Selected: radius 50 m, k = 10 — miss rate 0.0167, 6.83 candidates per observation, ceiling 0.9833.


![Candidate sweep](fig_candidate_sweep.png)


**Preprocessing and feature choices.** >>> WRITE THIS YOURSELF <<< — explain the per-trip local projection, the 100 m outlier cutoff, and why each feature group (geometric, directional, kinematic, topological) was included.


---

## 3. Model Experimentation and Comparison


### 3.1 Overall comparison

| model         | family             | config                                               |   cv_macro_f1 |   cv_top1 |   test_top1 |   test_macro_f1 |   test_log_loss |   test_route_continuity |   train_time_s |   inference_time_s |
|:--------------|:-------------------|:-----------------------------------------------------|--------------:|----------:|------------:|----------------:|----------------:|------------------------:|---------------:|-------------------:|
| random_forest | Ensemble (bagging) | 120 trees, min_samples_leaf=20, balanced_subsample   |        0.8921 |    0.823  |      0.8428 |          0.8991 |          0.1486 |                  0.835  |       159.214  |             0.5316 |
| knn           | Instance-based     | k=25, weights=distance, scaled, fit on 50k subsample |        0.8872 |    0.8095 |      0.8262 |          0.8967 |          0.2362 |                  0.8462 |         0.3728 |           403.588  |


![Model comparison](fig_model_comparison.png)


![Generalisation gap](fig_generalisation_gap.png)


![Accuracy vs cost](fig_accuracy_vs_cost.png)


### 3.2 Per-model discussion


#### random_forest (Ensemble (bagging))

- Configuration: `120 trees, min_samples_leaf=20, balanced_subsample`

- Student-test Top-1: **0.8428**, Macro F1 0.8991, Log Loss 0.1486

- Cross-validation Top-1: 0.8230 (gap to test +0.0197)

- Why suitable / strengths / limitations: >>> WRITE THIS YOURSELF <<<

- Overfitting or underfitting, with evidence: >>> WRITE THIS YOURSELF <<<


#### knn (Instance-based)

- Configuration: `k=25, weights=distance, scaled, fit on 50k subsample`

- Student-test Top-1: **0.8262**, Macro F1 0.8967, Log Loss 0.2362

- Cross-validation Top-1: 0.8095 (gap to test +0.0166)

- Why suitable / strengths / limitations: >>> WRITE THIS YOURSELF <<<

- Overfitting or underfitting, with evidence: >>> WRITE THIS YOURSELF <<<


---

## 4. Bayesian Hyperparameter Optimisation


Optuna TPESampler, 25 trials, group-aware cross-validation, objective = observation-level Top-1.


| variant   |   test_top1 |   test_macro_f1 |   test_log_loss |   fit_s |   inference_s |
|:----------|------------:|----------------:|----------------:|--------:|--------------:|
| untuned   |      0.8501 |          0.9144 |          0.1091 |    22.3 |          0.55 |
| tuned     |      0.8495 |          0.9156 |          0.1081 |   170.1 |          2.36 |


Tuned minus untuned Top-1: **-0.0006**.


**Interpretation.** >>> WRITE THIS YOURSELF <<< — the search did not improve Top-1 materially. Discuss why that is a legitimate result: the default configuration was already near-optimal for this feature set, the gain is within fold-to-fold noise, and the tuned model costs substantially more to train for no measurable benefit.


Full trial history: `results/optuna_trials.csv`.


---

## 5. HMM and Viterbi Route Reconstruction


### 5.1 Overall effect (observation-weighted, 10 student-test trips)

| Metric           |   Classifier only |   Classifier + HMM/Viterbi |   Change |
|:-----------------|------------------:|---------------------------:|---------:|
| Top-1 accuracy   |            0.8287 |                     0.8545 |   0.0258 |
| Route continuity |            0.8401 |                     0.9117 |   0.0716 |


Trips improved on Top-1: 9/10. On continuity: 10/10.


### 5.2 Per-trip results

|   trip_id |   n_obs |   clf_top1 |   hmm_top1 |   top1_delta |   clf_continuity |   hmm_continuity |   continuity_delta |   hmm_time_s |
|----------:|--------:|-----------:|-----------:|-------------:|-----------------:|-----------------:|-------------------:|-------------:|
|         0 |    2503 |     0.9057 |     0.9209 |       0.0152 |           0.8123 |           0.9228 |             0.1105 |        1.539 |
|         7 |    1354 |     0.9062 |     0.8508 |      -0.0554 |           0.9828 |           0.9841 |             0.0013 |        0.586 |
|         9 |    1049 |     0.836  |     0.8732 |       0.0372 |           0.7576 |           0.842  |             0.0844 |        2.28  |
|        40 |    2995 |     0.8972 |     0.9095 |       0.0124 |           0.913  |           0.9489 |             0.0359 |        2.816 |
|        52 |     262 |     0.9237 |     0.9656 |       0.042  |           0.7802 |           0.8588 |             0.0786 |        0.285 |
|        56 |    3958 |     0.8693 |     0.9155 |       0.0462 |           0.7597 |           0.8764 |             0.1168 |        5.773 |
|        65 |     768 |     0.7773 |     0.7891 |       0.0117 |           0.7903 |           0.8842 |             0.0939 |        3.969 |
|        76 |    1486 |     0.817  |     0.8627 |       0.0458 |           0.7296 |           0.8366 |             0.1069 |        9.83  |
|        77 |    3911 |     0.6689 |     0.7042 |       0.0353 |           0.9276 |           0.9574 |             0.0299 |       12.521 |
|        91 |     874 |     0.8135 |     0.8661 |       0.0526 |           0.7683 |           0.8462 |             0.0779 |        1.149 |


![HMM comparison](fig_hmm_comparison.png)


![HMM per-trip delta](fig_hmm_delta.png)


### 5.3 Where decoding did not help

|   trip_id |   max_gap_s |   clf_top1 |   hmm_top1 |   top1_delta |   clf_continuity |
|----------:|------------:|-----------:|-----------:|-------------:|-----------------:|
|         7 |           1 |     0.9062 |     0.8508 |      -0.0554 |           0.9828 |


>>> WRITE THIS YOURSELF <<< — explain this case. Note its classifier-only continuity was already very high, so there was little for the decoder to repair, and the transition prior pulled some correct independent predictions onto a smoother but wrong path.


**Hidden states, emissions, transitions, gap handling.** >>> WRITE THIS YOURSELF <<< — describe each, referring to `src/hmm.py`.


---

## 6. Geographical Visualisation

Interactive Folium map committed at `maps/trip_00000007.html` with four toggleable layers: raw GPS observations, road network, reference route and predicted route.


>>> WRITE THIS YOURSELF <<< — insert a screenshot and describe what it shows.


---

## 7. API, UI and Deployment

FastAPI backend on Laptop 1, Streamlit UI on Laptop 2. The UI lists the evaluation trips, lets the evaluator choose among the trained models and toggle HMM decoding, and renders the predicted route. Ground-truth labels are read server-side only; the response carries aggregate metrics alone.


>>> WRITE THIS YOURSELF <<< — insert screenshots of the working UI and of the API request logs on the receiving laptop.


---

## 8. Academic Integrity and AI-Use Disclosure

>>> WRITE THIS YOURSELF <<< — disclose AI assistance as required by the event policy, and confirm every result reported here is reproducible from the repository.
