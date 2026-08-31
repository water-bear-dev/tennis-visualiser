import json
import os
import streamlit as st
import pandas as pd
from PIL import Image
from src.config import (
    MATCH_SUMMARY_PATH,
    MATCH_REPORT_PATH,
    HEATMAP_P1_PATH,
    HEATMAP_P2_PATH,
    OUTPUT_PATH
)

# Page Configuration
st.set_page_config(
    page_title="AI Tennis Match Analytics & Coaching",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode & Sleek Cards)
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .kpi-card {
        background-color: #1a1f2c;
        border: 1px solid #30363d;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: bold; color: #f0f6fc; }
    .coaching-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎾 AI Tennis Visualiser & Coaching Intelligence")
st.markdown("Automated match tracking, kinematic stroke classification, and AI tactical coaching reports.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Match & Model Configuration")

uploaded_video = st.sidebar.file_uploader("Upload Match Footage (MP4)", type=["mp4"])
ball_conf = st.sidebar.slider("Ball Confidence Threshold", 0.05, 0.50, 0.08, 0.01)
person_conf = st.sidebar.slider("Player Confidence Threshold", 0.10, 0.80, 0.40, 0.05)
enable_roi = st.sidebar.checkbox("Enable Court ROI Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📁 Export & Downloads")

if os.path.exists(MATCH_SUMMARY_PATH):
    with open(MATCH_SUMMARY_PATH, "r") as f:
        json_data = f.read()
    st.sidebar.download_button(
        label="📥 Download match_summary.json",
        data=json_data,
        file_name="match_summary.json",
        mime="application/json"
    )

if os.path.exists(MATCH_REPORT_PATH):
    with open(MATCH_REPORT_PATH, "r") as f:
        html_data = f.read()
    st.sidebar.download_button(
        label="📄 Download HTML Report",
        data=html_data,
        file_name="match_report.html",
        mime="text/html"
    )

# --- Load Match Summary ---
match_data = {}
if os.path.exists(MATCH_SUMMARY_PATH):
    with open(MATCH_SUMMARY_PATH, "r") as f:
        match_data = json.load(f)

overview = match_data.get("match_overview", {})
ball_metrics = match_data.get("ball_metrics", {})
p1_metrics = match_data.get("player_1_near_court", {})
p2_metrics = match_data.get("player_2_far_court", {})
coaching = match_data.get("ai_coaching_intelligence", {})

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Match Rallies</div>
        <div class="kpi-value">{overview.get('total_rallies', '--')}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Longest Rally (Shots)</div>
        <div class="kpi-value">{overview.get('longest_rally_shots', '--')}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Peak Ball Shot Speed</div>
        <div class="kpi-value" style="color:#e3b341;">{ball_metrics.get('peak_shot_speed_kmh', '--')} km/h</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    tot_dist = (p1_metrics.get('total_distance_meters', 0.0) + p2_metrics.get('total_distance_meters', 0.0))
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Distance Ran</div>
        <div class="kpi-value">{tot_dist:.1f} m</div>
    </div>
    """, unsafe_allow_html=True)

# --- Tabbed Layout ---
tab_video, tab_coaching, tab_heatmaps = st.tabs(["📺 Match Video & Telemetry", "🧠 AI Coaching Insights", "🗺️ Court Heatmaps"])

with tab_video:
    vid_col, stats_col = st.columns([2, 1])
    with vid_col:
        if os.path.exists(OUTPUT_PATH):
            st.video(OUTPUT_PATH)
        elif os.path.exists("output.mp4"):
            st.video("output.mp4")
        else:
            st.info("Run `python main.py` to generate annotated video in `data/outputs/`.")

    with stats_col:
        st.markdown("### ⚔️ Player Head-to-Head")
        h2h_df = pd.DataFrame({
            "Metric": ["Distance Covered", "Peak Sprint Speed", "Avg Running Speed", "Strokes Played"],
            "Player 1 (Blue)": [
                f"{p1_metrics.get('total_distance_meters', 0.0)} m",
                f"{p1_metrics.get('peak_running_speed_kmh', 0.0)} km/h",
                f"{p1_metrics.get('average_running_speed_kmh', 0.0)} km/h",
                f"{p1_metrics.get('shots_played', 0)} shots"
            ],
            "Player 2 (Orange)": [
                f"{p2_metrics.get('total_distance_meters', 0.0)} m",
                f"{p2_metrics.get('peak_running_speed_kmh', 0.0)} km/h",
                f"{p2_metrics.get('average_running_speed_kmh', 0.0)} km/h",
                f"{p2_metrics.get('shots_played', 0)} shots"
            ]
        })
        st.table(h2h_df.set_index("Metric"))

with tab_coaching:
    st.subheader("🧠 Automated AI Coaching & Tactical Profiling")
    if coaching:
        st.info(f"**Match Tempo Profile:** {coaching.get('match_tempo_assessment', 'Standard Match')}")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("### 🟦 Player 1 Tactical Breakdown")
            st_p1 = coaching.get("stroke_breakdown", {}).get("player_1", {})
            st.write(f"**Stroke Distribution:** {st_p1.get('FOREHAND',0)} Forehands | {st_p1.get('BACKHAND',0)} Backhands | {st_p1.get('SERVE',0)} Serves | {st_p1.get('VOLLEY',0)} Volleys")
            for advice in coaching.get("player_1_tactical_insights", []):
                st.markdown(f"- {advice}")

        with c_col2:
            st.markdown("### 🟧 Player 2 Tactical Breakdown")
            st_p2 = coaching.get("stroke_breakdown", {}).get("player_2", {})
            st.write(f"**Stroke Distribution:** {st_p2.get('FOREHAND',0)} Forehands | {st_p2.get('BACKHAND',0)} Backhands | {st_p2.get('SERVE',0)} Serves | {st_p2.get('VOLLEY',0)} Volleys")
            for advice in coaching.get("player_2_tactical_insights", []):
                st.markdown(f"- {advice}")
    else:
        st.info("Run analysis to generate AI coaching intelligence.")

with tab_heatmaps:
    st.subheader("🗺️ 2D Spatial Court Heatmaps")
    hm_col1, hm_col2 = st.columns(2)
    with hm_col1:
        if os.path.exists(HEATMAP_P1_PATH):
            st.image(Image.open(HEATMAP_P1_PATH), caption="Player 1 Tactical Coverage (Near Court)", use_container_width=True)
    with hm_col2:
        if os.path.exists(HEATMAP_P2_PATH):
            st.image(Image.open(HEATMAP_P2_PATH), caption="Player 2 Tactical Coverage (Far Court)", use_container_width=True)
