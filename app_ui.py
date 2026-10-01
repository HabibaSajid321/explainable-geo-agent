"""
TrustGeoAgent - Streamlit Web Interface
----------------------------------------
Interactive web application for spatial model discovery, conformal uncertainty
quantification, spatial residual autocorrelation diagnostics, and LLM trust reporting.

Run with:
    streamlit run app_ui.py
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from data.synthetic import make_spatial_field, train_calib_test_split
from data.real_loader import load_real_field
from data.viz import export_interactive_map
from graph import build_graph

# Page Configuration
st.set_page_config(
    page_title="TrustGeoAgent - Trustworthy GeoAI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background: #1e293b; border-radius: 10px; padding: 12px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; font-size: 13px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 22px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 TrustGeoAgent — Trustworthy GeoAI Dashboard")
st.caption("Autonomous Multi-Agent Spatial Model Discovery, Conformal Uncertainty Quantification & Explainability")

# Sidebar
st.sidebar.header("🛠️ Pipeline Controls")
data_source = st.sidebar.radio("Data Source", ["Real Dataset (data/real_data.csv)", "Synthetic Spatial Benchmark", "Upload Custom CSV"])

custom_file = None
if data_source == "Upload Custom CSV":
    custom_file = st.sidebar.file_uploader("Upload CSV (lat, lon, value columns)", type=["csv"])

target_cov_pct = st.sidebar.slider("Target Coverage (%)", min_value=80, max_value=98, value=90, step=1)
seed = st.sidebar.number_input("Random Seed", value=0, step=1)
run_btn = st.sidebar.button("🚀 Run Agent Pipeline", type="primary", use_container_width=True)

# Helper function to run graph
def run_pipeline(coords, values, alpha):
    split = train_calib_test_split(coords, values, seed=seed)
    app = build_graph()
    result = app.invoke({"split": split, "interactive_review": False})
    return result

if run_btn or "last_result" in st.session_state:
    if run_btn:
        with st.spinner("Running Multi-Agent Discovery, Evaluation, Moran's I, and LLM Explainability..."):
            if data_source == "Upload Custom CSV" and custom_file is not None:
                df = pd.read_csv(custom_file)
                coords = df[["lat", "lon"]].to_numpy()
                values = df["value"].to_numpy()
            elif data_source == "Real Dataset (data/real_data.csv)":
                coords, values = load_real_field("data/real_data.csv")
            else:
                coords, values, _ = make_spatial_field(seed=seed)

            alpha = 1.0 - (target_cov_pct / 100.0)
            result = run_pipeline(coords, values, alpha)
            st.session_state["last_result"] = result
            st.session_state["data_source_name"] = data_source
    
    result = st.session_state["last_result"] = st.session_state.get("last_result")
    
    if result:
        disc = result["discovery_out"]
        ev = result["eval_out"]
        moran = ev.get("moran_res", {})
        scaling = ev.get("scaling_factor", 1.0)
        recal_attempts = result.get("recalibrate_attempts", 0)

        # Metrics Header Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Selected Model", disc["selected_model"].upper())
        m2.metric("Test RMSE", f"{ev['rmse']:.2f}")
        m3.metric("Coverage (Tgt / Emp)", f"{ev['target_coverage']*100:.0f}% / {ev['empirical_coverage']*100:.1f}%")
        m4.metric("Conformal Scale", f"{scaling:.3f}")
        m5.metric("Moran's I (Res)", f"{moran.get('morans_i', 0.0):.4f}")

        if recal_attempts > 0:
            st.success(f"⚡ LangGraph Autonomous Feedback Loop triggered {recal_attempts} conformal recalibration step(s) to align coverage.")

        # Tabs Layout
        tab_map, tab_llm, tab_cv, tab_audit = st.tabs(["🗺️ Interactive Spatial Map", "📄 Executive LLM Trust Report", "📊 Model Cross-Validation", "📜 Audit Log"])

        with tab_map:
            st.subheader("Spatial Uncertainty & Prediction Map")
            map_path = export_interactive_map(ev, selected_model=disc["selected_model"], output_path="spatial_trust_map.html")
            
            # Embed HTML Map safely
            if os.path.exists(map_path):
                with open(map_path, "r", encoding="utf-8") as f:
                    map_html = f.read()
                st.components.v1.html(map_html, height=600, scrolling=False)
            
            st.info("💡 **Green Markers**: Well-calibrated spatial predictions | **Red Markers**: Top quartile high-uncertainty predictions requiring human scrutiny.")

        with tab_llm:
            st.subheader("LLM Generated Trust & Calibration Assessment")
            st.markdown(result.get("explanation", "No explanation generated."))

        with tab_cv:
            st.subheader("Candidate Model Cross-Validation Scores")
            cv_df = pd.DataFrame(list(disc["scores"].items()), columns=["Spatial Model", "CV-RMSE"]).sort_values("CV-RMSE")
            st.dataframe(cv_df, use_container_width=True)
            st.write(f"**Selection Rationale:** {disc.get('rationale', 'N/A')}")

        with tab_audit:
            st.subheader("Append-Only Audit Log Trail")
            audit_file = "audit_log.jsonl"
            if os.path.exists(audit_file):
                with open(audit_file, "r") as f:
                    logs = [json.loads(line) for line in f if line.strip()]
                st.dataframe(pd.DataFrame(logs).iloc[::-1], use_container_width=True)
            else:
                st.write("No audit log found yet.")
else:
    st.info("👈 Click **Run Agent Pipeline** in the sidebar to run model discovery, uncertainty quantification, and generate the interactive map dashboard!")
