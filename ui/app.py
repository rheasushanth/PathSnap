"""PathSnap evaluator UI.  OWNER: D   Runs on LAPTOP 2.

Start with:
    streamlit run ui/app.py --server.port 8501

The API URL is a text input on purpose - on demo day you will need to repoint
it at Laptop 1's LAN address without touching code.
"""

import time

import folium
import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="PathSnap", page_icon="*", layout="wide")

st.title("PathSnap - Distributed Map-Matching & Route Reconstruction")
st.caption("UE24CS352A Machine Learning | Laptop 2 (UI) -> Laptop 1 (API)")

def normalise_url(raw: str) -> str:
    """Accept '10.30.202.245:8000', 'http://host:8000/', or a bare host.

    Typing the address without a scheme is the single most common demo-day
    mistake - requests then raises 'No scheme supplied' and it reads like a
    network failure when it is not. Default the port too, since the box is
    always pointed at our own uvicorn.
    """
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    if ":" not in raw.split("//", 1)[1]:
        raw += ":8000"
    return raw


default_url = st.session_state.get("api_url", "http://127.0.0.1:8000")
raw_url = st.text_input(
    "API base URL (Laptop 1)",
    value=default_url,
    help="e.g. http://10.30.202.245:8000 - scheme and port are filled in if omitted",
)
api_url = normalise_url(raw_url)
st.session_state["api_url"] = api_url or default_url

if not api_url:
    st.warning("Enter Laptop 1's address, e.g. http://10.30.202.245:8000")
    st.stop()
if api_url != raw_url.strip().rstrip("/"):
    st.caption(f"Using `{api_url}`")

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("Check connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            st.success(r.json())
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach {api_url}")
            st.info(
                "Work through these in order:\n\n"
                "1. On Laptop 1, is uvicorn running with `--host 0.0.0.0`? "
                "The default 127.0.0.1 only accepts local connections.\n"
                "2. Re-check Laptop 1's IP with `ipconfig` - it is DHCP and changes.\n"
                "3. Windows Firewall (admin PowerShell on Laptop 1):\n"
                "   `New-NetFirewallRule -DisplayName \"PathSnap API 8000\" "
                "-Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`\n"
                "4. Still failing? Campus Wi-Fi often blocks laptop-to-laptop "
                "traffic entirely (client isolation). Switch both laptops to a "
                "phone hotspot and use the new IP."
            )
        except requests.exceptions.Timeout:
            st.error("API reachable but did not respond in time.")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

# --- partition page ------------------------------------------------------
# One API process serves both partitions and picks per request, so switching
# to final evaluation needs no restart and no environment change on Laptop 1.
try:
    parts = requests.get(f"{api_url}/partitions", timeout=5).json()["partitions"]
except Exception:
    parts = []

page_names = {p["label"]: p for p in parts if p["available"]}
if page_names:
    page_label = st.sidebar.radio("Page", list(page_names))
    part = page_names[page_label]
    partition = part["name"]
    st.sidebar.caption(f"{part['n_trips']} trips")
    st.sidebar.info(part["note"])
else:
    partition = None
    st.sidebar.warning("No partitions reported by the API")

is_final = partition == "evaluator_test"
if is_final:
    st.subheader("Final Evaluation")
    st.warning(
        "Reserved evaluation partition. These trips were never used for "
        "training, feature selection, hyperparameter search or threshold "
        "choice. Ground-truth routes are read server-side only and are not "
        "returned to this interface.",
        icon=None,
    )
else:
    st.subheader("Development")

try:
    trip_params = {"partition": partition} if partition else {}
    trips = requests.get(f"{api_url}/trips", params=trip_params,
                         timeout=5).json()["trips"]
except Exception as e:
    st.warning(f"Could not load trip list: {e}")
    trips = []

if not trips:
    st.stop()

# Models the API can serve. Showing each one's measured score next to its
# name means the dropdown is self-explaining - the evaluator can see what
# they are choosing between without reading the report.
try:
    model_info = requests.get(f"{api_url}/models", timeout=5).json()
    model_list = model_info.get("models", [])
except Exception:
    model_list = []

labels = {f"{m['name']}  (Top-1 {m['student_test_top1']:.4f})": m["name"]
          for m in model_list}

with col_b:
    trip_id = st.selectbox("Evaluation trip", trips)

c1, c2 = st.columns([2, 1])
with c1:
    choice = st.selectbox(
        "Model", list(labels) or ["(API default)"],
        help="Scores shown are observation-level Top-1 on the 10 student-test trips",
    )
    model_name = labels.get(choice)
with c2:
    use_hmm = st.checkbox(
        "HMM / Viterbi decoding", value=True,
        help="Off = each GPS point classified independently. On = the whole "
             "sequence decoded so consecutive roads must actually connect.",
    )

# The full section 3.1 comparison table, served from the same generated CSV
# that goes into the report - so what the evaluator sees here and what they
# read on paper are the same numbers, not two transcriptions.
with st.expander("Model comparison — all models (Table 3.1)", expanded=False):
    try:
        comp = requests.get(f"{api_url}/comparison", timeout=5).json()
        if comp.get("rows"):
            st.dataframe(comp["rows"], hide_index=True, use_container_width=True)
            st.caption(
                "Student-test columns are the 10 held-out student-testing trips. "
                "CV Macro F1 is group-aware 5-fold cross-validation over the 80 "
                "training trips, grouped by trip so no trip appears in both folds."
            )
        else:
            st.info("Comparison table not generated yet — run `python -m src.report`.")
    except Exception as e:
        st.warning(f"Could not load comparison table: {e}")

if st.button("Run Prediction", type="primary"):
    t0 = time.time()
    params = {"hmm": str(use_hmm).lower()}
    if model_name:
        params["model"] = model_name
    if partition:
        params["partition"] = partition
    try:
        resp = requests.post(f"{api_url}/predict/{trip_id}", params=params, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-1 Accuracy", f"{m['top1_accuracy']:.3f}")
    c2.metric("Route Continuity", f"{m['route_continuity']:.3f}")
    c3.metric("Observations", m["n_observations"])
    c4.metric("Round trip", f"{time.time() - t0:.2f}s")

    st.caption(f"Model: **{result.get('model_name')}** "
               f"({result.get('model_family', 'n/a')})  |  "
               f"HMM decoding: **{'on' if result.get('hmm_enabled') else 'off'}**")

    # The whole thesis of the project in one table: what sequential decoding
    # bought on THIS trip, for both accuracy and physical plausibility.
    base = result.get("classifier_only", {})
    if base and result.get("hmm_enabled"):
        st.markdown("**Effect of HMM/Viterbi decoding on this trip**")
        st.dataframe([
            {"Metric": "Top-1 accuracy",
             "Classifier only": round(base.get("top1_accuracy", 0), 4),
             "+ HMM/Viterbi": round(m["top1_accuracy"], 4),
             "Change": round(m["top1_accuracy"] - base.get("top1_accuracy", 0), 4)},
            {"Metric": "Route continuity",
             "Classifier only": round(base.get("route_continuity", 0), 4),
             "+ HMM/Viterbi": round(m["route_continuity"], 4),
             "Change": round(m["route_continuity"] - base.get("route_continuity", 0), 4)},
        ], hide_index=True, use_container_width=True)
        if m.get("hmm_time_s"):
            st.caption(f"Viterbi decode took {m['hmm_time_s']:.2f}s of the "
                       f"{m['inference_time_s']:.2f}s total")

    if result.get("stage") == "passthrough":
        st.warning("API is in stage-1 passthrough mode - no model loaded yet. "
                   "The 'predicted route' shown is the raw GPS track.")

    gps = result["gps"]
    path = result["predicted_path"]
    if gps:
        centre = [sum(p[0] for p in gps) / len(gps), sum(p[1] for p in gps) / len(gps)]
        fmap = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap")

        g = folium.FeatureGroup(name="Raw GPS observations")
        for lat, lon in gps:
            folium.CircleMarker([lat, lon], radius=2, color="#1f77b4",
                                fill=True, fill_opacity=0.6).add_to(g)
        g.add_to(fmap)

        if path:
            p = folium.FeatureGroup(name="Predicted route")
            folium.PolyLine(path, color="#d62728", weight=4, opacity=0.8).add_to(p)
            p.add_to(fmap)

        folium.LayerControl().add_to(fmap)
        st_html(fmap._repr_html_(), height=600)

    with st.expander("Raw API response"):
        st.json({k: v for k, v in result.items() if k not in ("gps", "predicted_path")})
