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

try:
    trips = requests.get(f"{api_url}/trips", timeout=5).json()["trips"]
except Exception as e:
    st.warning(f"Could not load trip list: {e}")
    trips = []

if not trips:
    st.stop()

with col_b:
    trip_id = st.selectbox("Evaluation trip", trips)

if st.button("Run Prediction", type="primary"):
    t0 = time.time()
    try:
        resp = requests.post(f"{api_url}/predict/{trip_id}", timeout=120)
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

    base = result.get("classifier_only", {}).get("top1_accuracy")
    if base is not None:
        delta = m["top1_accuracy"] - base
        st.info(f"Classifier only: {base:.3f}  ->  Classifier + HMM/Viterbi: "
                f"{m['top1_accuracy']:.3f}   (delta {delta:+.3f})")

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
