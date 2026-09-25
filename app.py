"""
app.py
------
CyberSentinel AI — Unidirectional IDS dashboard.

This is the SIH-demo front end. It is a thin Streamlit layer over the real
trained ML module in src/detect.py (UnidirectionalIDS): a HistGradientBoosting
classifier trained on the CICIDS2017 dataset restricted to forward-direction
("unidirectional") flow features only.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from detect import UnidirectionalIDS          # noqa: E402
from config import FORWARD_FEATURES, MODELS_DIR, SAMPLE_DATA_DIR  # noqa: E402

st.set_page_config(page_title="CyberSentinel AI", page_icon="🛡️", layout="wide")

st.markdown(
    """<style>
    .block-container{padding-top:1.5rem;max-width:1400px}
    .metric-card{padding:16px;border:1px solid #ddd;border-radius:12px;background:#fff}
    .small{color:#666;font-size:13px}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ids():
    """Load the trained model/scaler/encoder once per session."""
    return UnidirectionalIDS(models_dir=MODELS_DIR)


@st.cache_data
def load_sample_data():
    return pd.read_csv(SAMPLE_DATA_DIR / "sample_flows.csv")


try:
    ids = get_ids()
    model_load_error = None
except FileNotFoundError as e:
    ids = None
    model_load_error = str(e)

st.title("🛡️ CyberSentinel AI")
st.caption(
    "AI-Based Detection of Cyber Threats in Unidirectional IP Traffic • SIH Prototype"
)

if model_load_error:
    st.error(
        "No trained model found in /models. Run this once from the project "
        "root before launching the app:\n\n"
        "```\ncd src\npython generate_sample_data.py --rows 8000\n"
        "python train.py --data-dir ../data/sample\n```"
    )
    st.stop()

with st.sidebar:
    st.header("Control Panel")
    source = st.radio("Traffic source", ["Demo traffic (sample_flows.csv)", "Upload CSV"], index=0)
    min_severity = st.select_slider(
        "Minimum alert severity",
        options=["none", "low", "medium", "high", "critical"],
        value="low",
    )
    st.info(
        "For SIH demonstration, start with Demo traffic. The same pipeline "
        "accepts real CICFlowMeter / forward-only flow CSV exports."
    )
    st.markdown("**Detection engine**")
    st.write("Trained on CICIDS2017 — HistGradientBoosting / Random Forest (best kept)")
    st.markdown("**Detected categories**")
    st.write("BENIGN • DoS • DDoS • PortScan • Brute Force • Botnet • Web Attack • Infiltration • Heartbleed")
    st.markdown("**Required input columns**")
    with st.expander("Show the 42 forward-direction features"):
        st.code("\n".join(FORWARD_FEATURES), language=None)

# ---------------------------------------------------------------------
# Load traffic
# ---------------------------------------------------------------------
raw = None
if source == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload a unidirectional flow CSV (CICFlowMeter-style forward features)",
        type=["csv"],
    )
    if uploaded:
        raw = pd.read_csv(uploaded)
        raw.columns = [c.strip() for c in raw.columns]
else:
    sample = load_sample_data()
    n = min(600, len(sample))
    raw = sample.sample(n=n, random_state=None).reset_index(drop=True)

# ---------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------
if raw is not None:
    missing = [f for f in FORWARD_FEATURES if f not in raw.columns]
    if missing:
        st.error(
            "Uploaded CSV is missing required forward-direction feature "
            f"columns: {missing[:8]}{'...' if len(missing) > 8 else ''}\n\n"
            "Expand 'Show the 42 forward-direction features' in the sidebar "
            "for the full required column list."
        )
        st.stop()

    result = ids.predict_batch(raw)

    total = len(result)
    attacks = int(result["is_attack"].sum())
    benign = total - attacks
    critical_high = int(result["severity"].isin(["critical", "high"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analyzed", f"{total:,}")
    c2.metric("Benign", f"{benign:,}")
    c3.metric("Threats detected", f"{attacks:,}", delta=f"{attacks/total*100:.1f}%" if total else "0%")
    c4.metric("Critical/High alerts", f"{critical_high:,}")

    st.divider()
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Threat category distribution")
        st.bar_chart(result["predicted_category"].value_counts())
    with right:
        st.subheader("Severity distribution")
        order = ["none", "low", "medium", "high", "critical"]
        counts = result["severity"].value_counts().reindex(order).fillna(0)
        st.bar_chart(counts)

    st.subheader("Recent detection events")
    display_cols = [c for c in ["Destination Port", "Flow Duration", "Total Fwd Packets"] if c in result.columns]
    display_cols += ["predicted_category", "confidence", "severity"]
    if "Label" in result.columns:
        display_cols = ["Label"] + display_cols  # show ground truth if present (demo data)
    view = result[display_cols].tail(25).iloc[::-1].copy()
    if "confidence" in view.columns:
        view["confidence"] = (view["confidence"] * 100).round(1)
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader(f"Alerts (severity ≥ {min_severity})")
    alerts = ids.alerts(result, min_severity=min_severity)
    st.dataframe(alerts, use_container_width=True, hide_index=True)
    st.caption(f"{len(alerts):,} of {total:,} flows raised an alert at this threshold.")

    st.download_button(
        "⬇️ Export full detection report (CSV)",
        result.to_csv(index=False).encode(),
        "cybersentinel_detection_report.csv",
        "text/csv",
    )

st.divider()
st.markdown(
    "**SIH note:** This prototype is for authorized lab/test traffic only. "
    "It detects and classifies patterns; it does not block traffic or attack systems."
)
