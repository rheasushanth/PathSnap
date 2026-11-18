# PathSnap — Team Execution Plan (4 people)

Hackathon: UE24CS352A Machine Learning, PES University RR Campus.
Phase-1 evaluation is **before lunch**. Final demo is a **two-laptop** live run.

Times below are relative to your start (`H+0`). Adjust once you know the real clock.

---

## 0. The three facts that drive every design decision

These came from inspecting the actual data, not the PDFs. Read them before writing code.

**Fact 1 — every trip has its own private road network.**
Bounding boxes of four trips:

| Trip | lon | lat | roughly |
|---|---|---|---|
| train/00000001 | 8.57–8.65 | 49.84–50.21 | Germany |
| train/00000002 | 11.00–11.05 | 47.66–47.70 | Austria / Bavaria |
| student_test/00000000 | 54.25–54.96 | 56.08–56.26 | Russia |
| student_test/00000052 | 35.86–35.90 | 56.80–56.83 | Russia |

Consequences:
- There is **no global graph**. `arc_id` only means something *inside one trip*.
- A multi-class classifier over arc IDs is impossible. Do not try.
- No single UTM zone works. Project **per trip**, locally.

**Fact 2 — `.route` is NOT per-observation labels.**
Trip `00000007`: 1354 GPS points, 2061 nodes, 4369 arcs, but only **123 route rows**.
`.route` is the chronological *arc sequence* the vehicle drove. You must **derive**
a per-observation label by aligning GPS points to that arc sequence. This is the
highest-risk step in the project — if it is wrong, every metric downstream is wrong.

**Fact 3 — therefore the model is a per-candidate binary ranker.**
One training row = one `(observation, candidate_arc)` pair.
Label = 1 if that candidate is the aligned true arc, else 0.
The predicted probability becomes the **HMM emission score**.
This generalises across trips and gives you Top-1 Accuracy and Candidate Miss Rate for free.

---

## 1. Roles

| Person | Owns | Never touches |
|---|---|---|
| **A** | Data loading, label alignment, EDA, Folium maps | models, api |
| **B** | Candidate generation, feature engineering, HMM + Viterbi | models, api |
| **C** | 5 models, Bayesian optimisation, all comparison plots | api, geometry |
| **D** | FastAPI, UI, two-laptop networking, repo hygiene, report assembly | model internals |

Everyone writes their own report section as they go. **Do not leave the report to the end.**

---

## 2. Repo layout — create this at H+0, before anything else

```
pathsnap/
├── README.md
├── requirements.txt
├── PLAN.md
├── data/                      # the extracted Student folder (gitignored if large)
│   ├── train/
│   └── student_test/
├── src/
│   ├── __init__.py
│   ├── io_utils.py            # A
│   ├── geo.py                 # A
│   ├── labels.py              # A
│   ├── candidates.py          # B
│   ├── features.py            # B
│   ├── hmm.py                 # B
│   ├── models.py              # C
│   ├── tune.py                # C
│   ├── metrics.py             # C
│   └── viz.py                 # A
├── notebooks/
│   ├── 01_eda.ipynb           # A
│   ├── 02_models.ipynb        # C
│   └── 03_hmm_eval.ipynb      # B
├── api/
│   └── main.py                # D
├── ui/
│   └── app.py                 # D
├── artifacts/                 # saved model, scaler, config json
├── results/                   # all CSVs, plots, optuna trials
└── maps/                      # folium html
```

**File ownership is strict.** One owner per file = no merge conflicts.
Branch per person: `feat/a-data`, `feat/b-candidates`, `feat/c-models`, `feat/d-api`.
Merge to `main` only at the sync points in §4.

### requirements.txt

```
numpy
pandas
scipy
scikit-learn
networkx
optuna
lightgbm
folium
matplotlib
seaborn
fastapi
uvicorn
requests
streamlit
joblib
```

---

## 3. Interface contracts — agree these at H+0, then work in parallel

Everybody codes against these signatures. Nobody waits for anybody.

```python
# src/io_utils.py  (A)
def load_trip(trip_dir: str) -> dict:
    """
    Returns:
      {
        'trip_id': str,
        'track':  np.ndarray (N,3)  # lon, lat, timestamp_seconds
        'nodes':  np.ndarray (M,2)  # lon, lat ; row index = node_id
        'arcs':   np.ndarray (K,2)  # from_node_id, to_node_id ; row index = arc_id
        'route':  np.ndarray (R,) | None   # chronological arc_id sequence
      }
    """

def list_trips(partition_dir: str) -> list[str]: ...


# src/geo.py  (A)
def project(lon, lat, lon0, lat0) -> (x, y):
    """Local equirectangular metres about (lon0, lat0). Vectorised."""

def point_to_segment(px, py, ax, ay, bx, by) -> (dist_m, t):
    """Perpendicular distance in metres, and t in [0,1] = position along segment."""


# src/labels.py  (A)
def align_route_to_track(trip: dict) -> np.ndarray:
    """Returns shape (N,) int array: true arc_id for each GPS observation, -1 if unassignable."""


# src/candidates.py  (B)
def generate_candidates(trip: dict, radius_m: float = 50.0, k: int = 10) -> pd.DataFrame:
    """
    One row per (observation, candidate arc). Columns:
      obs_idx, arc_id, dist_m, t_along
    """


# src/features.py  (B)
FEATURE_COLS: list[str]   # frozen list, in order

def build_features(trip: dict, cands: pd.DataFrame) -> pd.DataFrame:
    """Adds FEATURE_COLS to cands and returns it."""

def build_dataset(partition_dir: str, radius_m=50.0, k=10) -> pd.DataFrame:
    """
    Full training matrix across all trips. Columns:
      trip_id, obs_idx, arc_id, <FEATURE_COLS...>, label
    `trip_id` is the GroupKFold group. Save to results/dataset_train.parquet
    """


# src/hmm.py  (B)
def viterbi(trip, cands_with_proba: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Returns shape (N,) predicted arc_id per observation."""


# src/metrics.py  (C)
def evaluate(y_true, y_pred, proba=None) -> dict:
    """accuracy, precision, recall, macro_f1, log_loss"""

def top1_accuracy(df_with_proba) -> float: ...
def candidate_miss_rate(df) -> float: ...
def route_continuity(trip, pred_arc_ids) -> float:
    """Fraction of consecutive predicted-arc changes that are connected in the graph."""
```

**API contract (D codes against this from minute one):**

```
GET  /health                -> {"status": "ok"}
GET  /trips                 -> {"trips": ["00000003", ...]}   # the 10 evaluator trips
POST /predict/{trip_id}     -> {
    "trip_id": str,
    "gps":            [[lat, lon], ...],
    "predicted_path": [[lat, lon], ...],   # geometry of decoded arcs, in order
    "predicted_arcs": [int, ...],
    "metrics": {"top1_accuracy": float, "route_continuity": float,
                "n_observations": int, "inference_time_s": float},
    "classifier_only": {"top1_accuracy": float}   # for the HMM-helped comparison
}
```

Ground truth is read **server-side only** and never appears in the response beyond
aggregate metrics. No raw label arrays in the payload.

---

## 4. Timeline and sync points

| When | Milestone | Who |
|---|---|---|
| H+0:00 | Repo created, pushed, contracts agreed, everyone on their branch | all |
| H+0:30 | **Dummy API + UI talking across two laptops** | D |
| H+1:30 | `load_trip` + `align_route_to_track` working on one trip | A |
| H+2:00 | **SYNC 1** — merge A and B. Candidate miss rate measured. | A, B |
| H+3:00 | First dataset built, first model trained, baseline number exists | C |
| H+4:00 | **PHASE-1 CHECKPOINT** — EDA done, 2+ models trained, API live, Folium map exists | all |
| H+5:00 | 5 models done, comparison table + plots | C |
| H+6:00 | Viterbi beating classifier-only, measured | B |
| H+6:30 | **SYNC 2** — tuned model + HMM exported to `artifacts/` | B, C |
| H+7:30 | Real model wired into API, full two-laptop run on all 10 trips | D |
| H+8:30 | Report assembled, screenshots taken, repo cleaned | D + all |
| H+9:00 | Final demo | all |

---

## 5. Person A — Data, Labels, EDA, Maps

### A1. Loader (`src/io_utils.py`) — 20 min
All four file types are whitespace-separated with **no headers**.
```python
np.loadtxt(path)            # track, nodes
np.loadtxt(path, dtype=int) # arcs, route
```
- `.track` → 3 cols: **lon, lat, timestamp_seconds**
- `.nodes` → 2 cols: lon, lat. **Row index = node_id**
- `.arcs` → 2 cols: from_node_id, to_node_id. **Row index = arc_id**
- `.route` → 1 col: chronological arc_id sequence

Guard: `.route` may be missing for the evaluator partition. Return `None`, don't crash.
Verify arc endpoints are valid node indices; log any trip that fails.

### A2. Projection (`src/geo.py`) — 20 min
Trips span Germany to Russia, so degrees are not metres and the scale factor changes.
Use a **local equirectangular projection per trip**, anchored at the track centroid:

```python
R = 6371000.0
x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
y = R * np.radians(lat - lat0)
```

Accurate to well under a metre over a single trip's extent, and it is fast — which
matters because you will call it millions of times. Justify this choice in the report
(the rubric explicitly asks you to justify your coordinate system).

`point_to_segment` — standard projection of a point onto a segment, clamped to `[0,1]`:
```python
def point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denom = dx*dx + dy*dy
    t = np.where(denom > 0, ((px-ax)*dx + (py-ay)*dy) / np.maximum(denom, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t*dx, ay + t*dy
    return np.hypot(px-cx, py-cy), t
```

### A3. Label alignment (`src/labels.py`) — **90 min, the hard one**

Goal: turn a 123-arc route into 1354 per-observation labels.

Algorithm (monotonic dynamic programming):
1. Build the route's polyline: for route position `r` (arc `route[r]`), take the arc's
   two node coords, projected. Compute cumulative arc length along the route so each
   arc occupies an interval `[s_start[r], s_end[r]]` in metres.
2. For each observation `i` and each route position `r`, compute the perpendicular
   distance `D[i, r]` from GPS point `i` to arc `route[r]`.
   - Don't build the full `N × R` matrix blindly if it's large — restrict `r` to a
     sliding window around the previous best, width ~20. 1354×123 is fine as-is though.
3. DP with a monotonicity constraint (the vehicle cannot go backwards along its route):
   ```
   cost[i][r] = D[i][r] + min over r' <= r of ( cost[i-1][r'] + skip_penalty * (r - r') )
   ```
   `skip_penalty` ≈ 5.0 metres per skipped arc. Backtrack to get `r*` per observation.
4. Label for observation `i` = `route[r*[i]]`.
5. Set label to `-1` when `D[i][r*] > 100 m` (GPS outlier / tunnel). Report how many.

**Validation before you move on — do not skip this:**
- Fraction of observations labelled (expect >95%).
- Labels must be non-decreasing in route position by construction. Assert it.
- Median distance to assigned arc should be small (expect ~5–20 m). Print the histogram.
- Plot one trip: GPS points coloured by assigned arc. It should look like a clean
  progression along the road, not confetti.

If this looks wrong, **stop and fix it**. Everything downstream depends on it.

### A4. EDA + report §2 — 60 min, `notebooks/01_eda.ipynb`
The rubric wants all of these. Produce each as a saved figure in `results/`:
- Dataset inventory table: trips per partition, observations per trip, file sizes.
- Observations-per-trip histogram; trajectory duration; total distance.
- Sampling interval distribution — confirm the "~97% are exactly 1 s" claim yourself
  and report the actual number. Count gaps > 5 s and flag which trips have them.
- Missing / duplicate / non-monotonic timestamps; duplicate coordinates.
- Geographic extent per trip (the table in §0 — reproduce it for all 90 trips).
- Road network size per trip: nodes, arcs, mean degree.
- Connectivity via NetworkX: number of connected components, isolated nodes.
- Label-quality stats from A3.

### A5. Folium (`src/viz.py`) — 40 min
`plot_trip(trip, pred_arcs=None, out_html=...)` with **layer controls**:
- Layer 1: raw GPS points (blue CircleMarkers)
- Layer 2: road network arcs (thin grey PolyLines) — subsample if the trip is huge
- Layer 3: reference route (green PolyLine)
- Layer 4: predicted route (red PolyLine)
- Legend, sensible zoom, `folium.LayerControl()`

**Coordinate order trap:** files are `lon, lat`. Folium takes `[lat, lon]`. Flipping
these is the single most common bug in this project — if your map shows the Indian
Ocean or Antarctica, this is why.

Save to `maps/trip_<id>.html`, commit it, screenshot it for the PDF.

---

## 6. Person B — Candidates, Features, HMM

### B1. Candidate generation (`src/candidates.py`) — 50 min
1. Project all nodes once per trip.
2. Build arc midpoints; `scipy.spatial.cKDTree` over them.
3. For each observation, query the nearest ~40 arc midpoints, then compute **exact**
   `point_to_segment` distance to those arcs, keep the ones within `radius_m`, take the
   top `k` by true distance.
   - Midpoint pre-filter with a generous `k` then exact re-rank. Querying midpoints alone
     is wrong for long arcs; the re-rank fixes it.
4. If zero candidates found, fall back to the single nearest arc regardless of radius —
   never emit an empty candidate set, the Viterbi needs at least one state per step.

**Immediately tune this against Candidate Miss Rate**, which is a hard ceiling on your
final accuracy. Sweep and save the table to `results/candidate_sweep.csv`:

| radius_m | k | miss_rate | mean candidates/obs |
|---|---|---|---|
| 30 | 5 | ? | ? |
| 50 | 10 | ? | ? |
| 80 | 15 | ? | ? |
| 120 | 20 | ? | ? |

Pick the smallest radius/k with miss rate below ~2%. Larger `k` means a slower Viterbi
(cost is `O(N·k²)`), so don't just max it out. This table goes straight into the report.

### B2. Features (`src/features.py`) — 60 min
Per `(observation, candidate)` row. Freeze `FEATURE_COLS` early and tell C immediately.

Geometric:
- `dist_m` — perpendicular distance to the arc
- `dist_rank` — 0-based rank among this observation's candidates
- `dist_ratio` — `dist_m / (min_dist_for_this_obs + 1e-6)`
- `dist_gap` — `dist_m - min_dist_for_this_obs`
- `t_along` — position along the arc, `[0,1]`
- `arc_length_m`

Directional:
- `gps_bearing` — bearing from observation `i-1` to `i+1` (smoothed; handles noise better
  than consecutive points)
- `arc_bearing` — bearing of the arc
- `bearing_diff` — `abs` angular difference folded into `[0, 180]`
- `cos_bearing_diff` — a smoother version of the same signal, often more useful to linear models

Kinematic:
- `speed_mps` — from consecutive observations
- `dt_prev`, `dt_next` — timestamp deltas (catches the gap trips)

Topological (NetworkX graph of the trip, built once):
- `from_degree`, `to_degree` — node degrees of the arc's endpoints
- `connected_to_prev_best` — 1 if this arc shares a node with, or equals, the previous
  observation's nearest candidate. Cheap sequential hint; it measurably helps.

Context:
- `n_candidates` for this observation

**Do not** include `trip_id`, `arc_id`, `obs_idx`, raw lon/lat, or anything absolute —
those don't transfer across trips and will leak or mislead.

`build_dataset` loops all 80 training trips, concatenates, saves parquet. Expect roughly
80 trips × ~2000 obs × ~8 candidates ≈ 1.3M rows. Fine in memory as float32.

### B3. HMM + Viterbi (`src/hmm.py`) — 90 min
Newson & Krumm formulation, in log space throughout.

- **Hidden states at step `i`** = the candidate arcs for observation `i`.
- **Emission:** `log(classifier_proba + 1e-12)`. The classifier's calibrated probability
  that this candidate is correct *is* the emission. Say exactly this in the report —
  it's the question they ask.
- **Transition** from candidate `u` at step `i-1` to candidate `v` at step `i`:
  ```
  d_gps  = great-circle distance between observation i-1 and i
  d_road = network shortest-path distance between arc u and arc v
  delta  = abs(d_gps - d_road)
  log P  = -delta / beta        # exponential, beta ≈ 10-30 m (tune it)
  ```
  Compute `d_road` with a **Dijkstra cutoff** (e.g. `3 * d_gps + 100 m`) on a node graph
  weighted by arc length — not all-pairs, that will be far too slow.
  Cache per `(u, v)` pair within a trip.
- **Disconnected / unreachable:** apply a large finite penalty (e.g. `-20.0`), **never
  `-inf`**. A `-inf` row makes the whole path infeasible and your decode collapses.
- **GPS gaps:** if `dt > 30 s` or `d_gps > 500 m`, soften the transition toward uniform —
  you have no information about what happened in between, so don't pretend you do.
- **Viterbi:** standard `O(N·k²)` forward pass storing backpointers, then backtrack.

**Tuning split — read this, the obvious approach is wrong.**
Tuning `beta` on all 10 student-test trips and then *reporting* on those same 10
trips makes your HMM numbers optimistic: you fitted the decoder to the set you're
scoring on. Instead do one of:

- **Preferred:** tune `beta` and the penalty by group-CV on the **80 training
  trips**, then report on all 10 student-test trips, which stay genuinely held out.
- **Acceptable:** split student-test 5 tune / 5 report, and say so in the report.

Either way, state explicitly in §5 which trips tuned the decoder and which produced
the reported numbers. Evaluators look for this.

Save the chosen values to `artifacts/hmm_config.json`.

### B4. Evaluation + report §5 — 40 min, `notebooks/03_hmm_eval.ipynb`
Produce this table over the 10 student-test trips:

| | Top-1 Acc | Macro F1 | Route Continuity |
|---|---|---|---|
| Classifier only (argmax per obs) | | |
| Classifier + HMM/Viterbi | | |

Then write up: which trips improved, which got worse, and **why**. Honest discussion of
failure cases scores better than claiming it always helped. Typical finding: HMM gives a
large continuity gain and a modest accuracy gain, and occasionally hurts on trips with
big GPS gaps. Say so if that's what you see.

---

## 7. Person C — Models, Tuning, Plots

### C1. Splitting — 10 min, get this right first
```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=df['trip_id']):
    ...
```
**Group by `trip_id`.** A random split puts observations from the same trip on both
sides and inflates your scores enormously. The rubric calls this out explicitly and the
viva will ask.

Fit the scaler **inside** the fold, on train only.

### C2. Train five models — 90 min, `notebooks/02_models.ipynb`
Deliberately spread across algorithm families (the rubric rewards the spread):

| # | Model | Family | Key config |
|---|---|---|---|
| 1 | LogisticRegression | Linear / probabilistic | `class_weight='balanced'`, scaled |
| 2 | KNeighborsClassifier | Instance-based | `n_neighbors=25`, `weights='distance'`, scaled |
| 3 | RandomForestClassifier | Ensemble | `n_estimators=300`, `min_samples_leaf=5` |
| 4 | LightGBM / HistGradientBoosting | Boosting | `n_estimators=400`, `learning_rate=0.05` |
| 5 | MLPClassifier | Neural network | `hidden_layer_sizes=(64,32)`, `early_stopping=True` |

For every model record, into `results/model_comparison.csv`:
`cv_macro_f1`, `student_test_macro_f1`, `top1_accuracy`, `route_continuity`,
`log_loss`, `precision`, `recall`, `train_time_s`, `inference_time_s`.

Set `random_state=42` everywhere and record it — reproducibility is marked.

**Note on class imbalance:** with k=10 candidates, roughly 1 positive per 10 rows.
Use `class_weight='balanced'` where supported and report **macro** F1, not accuracy —
plain accuracy will sit around 0.9 while predicting all-negative and tell you nothing.

**Top-1 accuracy is computed differently from classification accuracy.** Group the
predicted probabilities by `(trip_id, obs_idx)`, take the `argmax` candidate, and check
whether its `arc_id` equals the true label. That is the metric that actually matters here —
make it your primary selection metric and say why in the report.

### C3. Bayesian optimisation (`src/tune.py`) — 60 min
Optuna, on whichever model won C2 (usually the boosting one).

```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
study.trials_dataframe().to_csv('results/optuna_trials.csv')
```
- Objective: group-aware CV **Top-1 accuracy** (or macro F1 — pick one, justify it).
- Search space, with justification for each range in the report.
  For LightGBM: `num_leaves` 15–255, `learning_rate` 0.01–0.3 (log),
  `n_estimators` 100–1000, `min_child_samples` 5–100, `subsample` 0.6–1.0,
  `colsample_bytree` 0.6–1.0, `reg_lambda` 1e-3–10 (log).
- 30–50 trials. Save the trials dataframe — **the rubric requires the search record.**
- Refit on all 80 training trips with the best params.
- Export: `joblib.dump({'model': m, 'scaler': s, 'features': FEATURE_COLS}, 'artifacts/model.pkl')`

### C4. Plots — 40 min, every one from real results
Save to `results/`:
- CV macro-F1 bar chart across all 5 models
- Student-test Top-1 accuracy + macro F1 grouped bar chart
- Log loss comparison
- Confusion matrix for the final model
- Optuna optimisation history (`optuna.visualization.matplotlib.plot_optimization_history`)
- Optuna param importances
- Feature importance for the final model
- Tuned vs untuned side-by-side

**Do not fabricate training curves** for models that have no epochs. The rubric says this
twice. Only the MLP has a `loss_curve_`; plot it only for the MLP.

### C5. Report §3 and §4 — 50 min
§3.2 demands a **separate subsection per model**, each answering:
1. Why is this algorithm suitable for PathSnap?
2. Major strengths and limitations *for this task*?
3. Important hyperparameters and initial values?
4. Loss function / optimisation objective?
5. Performance on student-test?
6. Top-1 accuracy, macro F1, log loss on student-test?
7. Overfitting or underfitting — **with evidence** (train vs CV gap)?

Answer from *your* numbers. Textbook descriptions get marked down explicitly.

---

## 8. Person D — API, UI, Integration, Report

### D1. Dummy API first — do this at H+0:30, before any model exists

This is the most important scheduling decision in the plan. Integration is what kills
teams, not modelling. Prove the two-laptop path works while everything else is still
being written.

`api/main.py`: implement every endpoint in §3, returning the trip's **raw GPS points**
as the "predicted path" and zeros for metrics. Structure it so the only thing that
changes later is one function:

```python
def predict_route(trip_id: str) -> dict:
    # H+0:30  -> return raw GPS as the path
    # H+7:00  -> load artifacts/model.pkl, run candidates -> features -> proba -> viterbi
```

Run with:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` is mandatory — the default `127.0.0.1` is unreachable from Laptop 2.

### D2. Two-laptop networking — 20 min, and test it NOW
On Laptop 1:
```bash
ipconfig
```
Note the IPv4 address (e.g. `192.168.1.37`). From Laptop 2:
```bash
curl http://192.168.1.37:8000/health
```
If it fails, it is almost always Windows Firewall. Allow inbound TCP 8000 for Python,
or switch both laptops to a phone hotspot. **Sort this out in the morning, not at 4pm.**
Have the hotspot as a tested fallback.

Enable CORS in FastAPI or the browser UI will be blocked:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

### D3. UI — 60 min
Streamlit is the fastest route:
- Text input for the API base URL (so you can repoint it without a code change — do this,
  you will need it)
- `GET /trips` → dropdown of the 10 evaluator trips
- "Run Prediction" button → `POST /predict/{trip_id}`
- Render the returned Folium map via `streamlit.components.v1.html`
- Metrics panel: Top-1 accuracy, route continuity, inference time, observation count
- Show classifier-only vs classifier+HMM side by side — it demos the whole thesis in one view

```bash
streamlit run ui/app.py --server.port 8501
```

### D4. Wire in the real model — 40 min at H+7:00
Swap `predict_route` to the real pipeline. Load the model **once at startup**, not per
request. Add request logging so the evaluator can see live traffic:

```python
@app.post("/predict/{trip_id}")
async def predict(trip_id: str):
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] REQUEST trip={trip_id}")
    result = predict_route(trip_id)
    print(f"[{datetime.now():%H:%M:%S}] DONE trip={trip_id} "
          f"top1={result['metrics']['top1_accuracy']:.3f} "
          f"t={time.time()-t0:.2f}s")
    return result
```

The rubric requires showing the evaluator the API logs on the receiving laptop. Make them
readable and keep that terminal window visible during the demo.

**Then test all ten evaluator trips.** The rubric says it must work for all ten, not one.
Loop them from a script and confirm none throws.

### D5. Repo hygiene — 30 min
- `README.md`: setup, install, how to run API, how to run UI, how to reproduce training,
  folder map, team member names
- `.gitignore`: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, large data
- **No secrets committed.** No API keys, no passwords.
- Commit `artifacts/model.pkl` and `artifacts/hmm_config.json` if they're reasonably sized
- Commit the Folium HTML maps
- Make the repo **public** and verify in an incognito window that it actually loads
- Commit steadily all day — repository history is reviewed, and a single giant end-of-day
  commit looks bad

### D6. Report assembly — 60 min
One consolidated PDF. Sections per the rubric: §2 EDA (A), §3 models (C), §4 Bayesian
optimisation (C), §5 HMM (B), §6 Folium maps (A), §7 API/UI with screenshots (D).

Screenshots needed:
- Working UI with a predicted route rendered
- API terminal showing successful request logs
- At least one Folium map

Every number in the PDF must be reproducible from the repo. They cross-check.

---

## 9. Pre-demo checklist

- [ ] API runs on Laptop 1, bound to `0.0.0.0`, terminal visible
- [ ] UI runs on Laptop 2 and reaches the API over the network
- [ ] All 10 evaluator trips tested end to end, none crash
- [ ] Ground truth never leaves the server beyond aggregate metrics
- [ ] Zero hardcoded predictions anywhere (they will grep for it)
- [ ] GitHub public, README complete, verified in incognito
- [ ] PDF exported, every figure from real results
- [ ] `results/optuna_trials.csv` committed
- [ ] `results/model_comparison.csv` committed with all 5 models
- [ ] Folium HTML committed
- [ ] Each person can explain **someone else's** section — they ask cross-questions

---

## 10. Traps that will cost you marks

1. **lon/lat swapped.** Files are `lon, lat`; Folium wants `[lat, lon]`. Map in the ocean = this.
2. **Random split instead of GroupKFold by trip.** Inflates every score. Explicitly penalised.
3. **Euclidean distance in degrees.** Wrong by a factor that varies with latitude, and these
   trips span 44°N to 56°N. The rubric asks you to justify your projection.
4. **Treating `.route` as per-observation labels.** 123 rows vs 1354 observations.
5. **A global arc-ID classifier.** Arc IDs are per-trip. Doesn't exist.
6. **`-inf` transitions in Viterbi.** Use a large finite penalty.
7. **Reporting plain accuracy on 10:1 imbalanced candidate rows.** Use macro F1 and Top-1.
8. **Touching evaluator-test data.** Not for training, tuning, feature selection, or thresholds.
9. **Fabricated training curves** for models without epochs. Called out twice in the rubric.
10. **Leaving the report to the end.** It is 50% of the deliverable. Write sections as you finish them.
11. **`--host 127.0.0.1`** on the API. Laptop 2 cannot reach it.
12. **One giant commit at 5pm.** Repository history is part of the integrity review.

---

## 11. If you fall behind — cut in this order

Protect the demo and the report; sacrifice depth.

1. Drop `k` to 5 and radius to 40 m — faster everything, small accuracy cost
2. Train on a 40-trip subset instead of 80
3. Optuna 15 trials instead of 40 (but **never zero** — Bayesian optimisation is mandatory)
4. Skip the MLP, use a Decision Tree (still five models, still five families)
5. Simplify Viterbi transitions to a connectivity check instead of Dijkstra distance
   (much faster, still a legitimate HMM — just document the simplification honestly)

**Never cut:** five models, Bayesian optimisation, HMM/Viterbi, the Folium map, the
two-laptop demo, the PDF. Those are each separately mandatory, and the policy states a
high score on one metric does not compensate for a missing component.

---

## 12. Differentiation — what lifts this above a correct-but-standard submission

Be honest with yourselves about the baseline: §5–§8 is the **Newson & Krumm (2009)**
HMM map-matching framework with a learned emission model. That is the canonical
approach, it is the right choice, and roughly every competent team will land near it.
Rigour alone will beat most of them. These additions are what separate first from third.

Reference for the cleaning and confidence ideas: Zepto's *Route IQ* engineering blog
(blog.zepto.com, Aug 2026), which solves this exact problem for delivery fleets.
**Cite it in the report.** Adapting a documented industrial technique and measuring it
on your data is good engineering. Presenting it as your own invention is not.

### Ranked by return on time

| # | Addition | Time | Why it pays |
|---|---|---|---|
| 1 | **Emission calibration** (`models.calibrate`) | 30 m | Fixes a real bug, not just novelty |
| 2 | **GPS cleaning pipeline** (`src/clean.py`) | 90 m | Biggest visible differentiator; answers "how did you preprocess?" |
| 3 | **Feature-group ablation** (`models.ablate_feature_groups`) | 30 m | Analytical depth, reuses existing pipeline |
| 4 | **Adaptive beta** (`hmm.adaptive_beta`) | 20 m | Departs from textbook, defensible |
| 5 | **Confidence scoring + fallback** (`hmm.confidence_label`) | 40 m | Engineering maturity; great UI demo |

**#1 is a correctness fix, not a nice-to-have.** Three of the five models use
`class_weight='balanced'`, which deliberately distorts output probabilities, and tree
ensembles are badly calibrated by default. Those probabilities *are* the HMM emissions.
Uncalibrated, Viterbi is balancing a skewed emission term against a well-specified
transition term and quietly optimising the wrong objective. Do this one even if you
do nothing else here.

**#2 is the most visible.** Most teams feed raw GPS straight into candidate generation.
A real cleaning stage — multi-factor anomaly scoring, iterative one-at-a-time removal,
stall compression, Kalman smoothing — is immediately legible to an evaluator and
directly answers a rubric question. See `src/clean.py` for the method.

### Every addition must come with a measured ablation

This is the whole point. Each one gets a row in a table with the same metric, same
trips, feature on vs off:

| Variant | Top-1 Acc | Macro F1 | Route Continuity |
|---|---|---|---|
| Classifier only | | | |
| + HMM/Viterbi (fixed beta, uncalibrated) | | | |
| + calibrated emissions | | | |
| + GPS cleaning | | | |
| + adaptive beta | | | |

**Report negative results honestly.** If cleaning hurts accuracy, say so and reason
about why — the rubric explicitly asks you to "discuss cases where HMM decoding did not
help." A team that reports a feature made things 2% worse and explains the mechanism
demonstrates more understanding than one whose every number happens to go up.

### Deliberately out of scope

Not worth the time today, but good answers if the viva asks "what would you do next":
a learned transition model, a GNN over the arc-adjacency graph, a sequence model for
temporal context beyond the single `connected_to_prev_best` feature, and online
streaming Viterbi with fixed lag (which is what a real navigation app actually needs).
