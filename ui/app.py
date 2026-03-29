"""
🛰️ AI Landslide Monitoring System — Production Dashboard
============================================================
Enterprise-grade real-time monitoring dashboard for landslide
prediction and detection across Indian terrain stations.

Features:
  • Real-time weather data from Open-Meteo API
  • Extended climate metrics (wind, humidity, pressure, visibility)
  • Multi-factor risk prediction with ML-powered early warnings
  • YOLOv11 satellite imagery analysis
  • 60+ Indian hill station coverage
  • Event reporting and alert system

Version: 3.0.0 (Production)
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime, timedelta

import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction.weather_data import (
    WeatherDataFetcher, WeatherRiskCalculator, MONITORING_STATIONS
)
from detection.detect_landslides import LandslideDetector
from prediction.risk_estimator import RiskEstimator
from prediction.live_satellite import LiveSatelliteFetcher
from visualization.draw_boxes import Visualizer
from preprocessing.preprocess import ImagePreprocessor

# Indian Hills satellite metadata
INDIAN_HILLS_DIR = os.path.join(PROJECT_ROOT, "dataset", "indian_hills")
try:
    _hills_meta_path = os.path.join(INDIAN_HILLS_DIR, "metadata.json")
    if os.path.isfile(_hills_meta_path):
        with open(_hills_meta_path) as _f:
            INDIAN_HILLS_META = json.load(_f)
    else:
        INDIAN_HILLS_META = {}
except Exception:
    INDIAN_HILLS_META = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Landslide Monitoring System | NDMA India",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://ndma.gov.in/',
        'Report a bug': None,
        'About': "🛰️ AI-Powered Landslide Monitoring System v3.0\n\nReal-time monitoring for 60+ Indian hill stations using satellite imagery and weather data."
    }
)

# ════════════════════════════════════════════════════════════════════
# PRODUCTION CSS — Modern Dark Theme
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ═══════════ GLOBAL STYLES ═══════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card: #16213e;
    --bg-elevated: #1f2940;
    --accent-primary: #00d9ff;
    --accent-secondary: #7b2cbf;
    --accent-gradient: linear-gradient(135deg, #00d9ff 0%, #7b2cbf 100%);
    --text-primary: #ffffff;
    --text-secondary: #a0aec0;
    --text-muted: #718096;
    --border-subtle: rgba(255,255,255,0.08);
    --shadow-glow: 0 0 40px rgba(0,217,255,0.15);
    --safe: #00e676;
    --watch: #ffc107;
    --warning: #ff9800;
    --danger: #f44336;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(180deg, var(--bg-primary) 0%, #0a0a14 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ═══════════ HEADER BAR ═══════════ */
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow-glow), 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}

.header-left h1 {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.header-left .tagline {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 4px;
    font-weight: 400;
}

.header-right {
    text-align: right;
}

.live-indicator {
    display: inline-flex;
    align-items: center;
    background: rgba(0,230,118,0.15);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--safe);
    border: 1px solid rgba(0,230,118,0.3);
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--safe);
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse-live 2s ease-in-out infinite;
    box-shadow: 0 0 10px var(--safe);
}

@keyframes pulse-live {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

.station-info {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 8px;
}

.station-info strong {
    color: var(--text-primary);
}

/* ═══════════ ALERT BANNER ═══════════ */
.alert-banner {
    padding: 1rem 1.5rem;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 1.5rem;
    letter-spacing: 0.3px;
    border: 1px solid transparent;
    position: relative;
    overflow: hidden;
}

.alert-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    opacity: 0.1;
    background: radial-gradient(ellipse at center, white 0%, transparent 70%);
}

.alert-safe {
    background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
    color: #fff;
    box-shadow: 0 4px 20px rgba(0,200,83,0.3);
}

.alert-watch {
    background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%);
    color: #1a1a2e;
    box-shadow: 0 4px 20px rgba(255,152,0,0.3);
}

.alert-warning {
    background: linear-gradient(135deg, #f44336 0%, #ff5722 100%);
    color: #fff;
    box-shadow: 0 4px 20px rgba(244,67,54,0.4);
}

.alert-danger {
    background: linear-gradient(135deg, #b71c1c 0%, #d32f2f 100%);
    color: #fff;
    box-shadow: 0 0 40px rgba(183,28,28,0.5);
    animation: danger-pulse 1.5s ease-in-out infinite;
}

@keyframes danger-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(183,28,28,0.4); }
    50% { box-shadow: 0 0 50px rgba(183,28,28,0.7); }
}

/* ═══════════ METRIC CARDS ═══════════ */
.metric-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    min-height: 130px;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    border-color: rgba(0,217,255,0.3);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent-gradient);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-card:hover::before {
    opacity: 1;
}

.mc-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
}

.mc-value {
    font-size: 2rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 4px 0;
}

.mc-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

/* ═══════════ SECTION HEADERS ═══════════ */
.section-header {
    display: flex;
    align-items: center;
    margin: 2rem 0 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--border-subtle);
}

.section-header h2 {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header .badge {
    font-size: 0.7rem;
    background: var(--accent-gradient);
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ═══════════ RISK SCORE DISPLAY ═══════════ */
.risk-score-container {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.risk-score-value {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin: 0.5rem 0;
    text-shadow: 0 4px 20px currentColor;
}

.risk-score-label {
    font-size: 0.9rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.risk-level-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 1rem;
}

/* ═══════════ RISK FACTOR BARS ═══════════ */
.risk-factor {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.risk-factor:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(0,217,255,0.2);
}

.rf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.rf-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
}

.rf-value {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.rf-value strong {
    color: var(--text-primary);
}

.rf-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
}

.rf-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease-out;
}

/* ═══════════ SOIL MOISTURE CARD ═══════════ */
.soil-card {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.soil-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

.soil-depth {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 8px;
}

.soil-value {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}

/* ═══════════ DETECTION RESULTS ═══════════ */
.detection-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.detection-card:hover {
    background: rgba(255,255,255,0.05);
}

.detection-card.landslide { border-left-color: #f44336; }
.detection-card.debris { border-left-color: #ff9800; }
.detection-card.normal { border-left-color: #4caf50; }

/* ═══════════ CHARTS ═══════════ */
.chart-container {
    background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ═══════════ EVENT LOG ═══════════ */
.event-log-item {
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.9rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.event-log-item:last-child {
    border-bottom: none;
}

.event-time {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 500;
    min-width: 70px;
}

/* ═══════════ FOOTER ═══════════ */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border-subtle);
}

.footer p {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin: 4px 0;
}

.footer .brand {
    font-size: 0.9rem;
    font-weight: 600;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ═══════════ SIDEBAR STYLING ═══════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    color: var(--text-primary);
}

/* ═══════════ BUTTON OVERRIDES ═══════════ */
.stButton > button {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(0,217,255,0.25) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,217,255,0.4) !important;
}

/* ═══════════ SELECT BOX STYLING ═══════════ */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
    border-radius: 10px !important;
}

/* ═══════════ TABS ═══════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-gradient);
    color: white;
}

/* ═══════════ INFO BOXES ═══════════ */
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* ═══════════ RESPONSIVE ═══════════ */
@media (max-width: 768px) {
    .header-bar {
        flex-direction: column;
        text-align: center;
        gap: 1rem;
    }
    .header-right {
        text-align: center;
    }
    .metric-card {
        min-height: 110px;
    }
    .mc-value {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# INITIALIZE COMPONENTS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_detector():
    return LandslideDetector(demo_mode=False)

@st.cache_resource
def init_tools():
    return ImagePreprocessor(target_size=640), RiskEstimator(), Visualizer()

detector = init_detector()
preprocessor, risk_estimator, visualizer = init_tools()


# ════════════════════════════════════════════════════════════════════
# SIDEBAR — Control Panel
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🛰️</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00d9ff;">Control Center</div>
        <div style="font-size: 0.75rem; color: #718096; margin-top: 4px;">AI Landslide Monitoring</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Region filter
    st.markdown("##### 🗺️ Region Filter")
    all_regions = sorted(set(s["region"] for s in MONITORING_STATIONS.values()))
    selected_region = st.selectbox(
        "Select Region",
        ["All Regions"] + all_regions,
        index=0,
        help="Filter stations by geographic region"
    )
    
    # Filter stations by region
    if selected_region == "All Regions":
        filtered_stations = MONITORING_STATIONS
    else:
        filtered_stations = {k: v for k, v in MONITORING_STATIONS.items() if v["region"] == selected_region}
    
    st.markdown("##### 📍 Monitoring Station")
    station_names = list(filtered_stations.keys())
    selected_station = st.selectbox(
        "Select Location",
        station_names,
        index=0,
        help="Choose an Indian hill station to monitor"
    )
    station_info = MONITORING_STATIONS[selected_station]
    
    st.markdown(f"""
    <div style="background: rgba(0,217,255,0.1); padding: 12px; border-radius: 10px; margin-top: 8px; border: 1px solid rgba(0,217,255,0.2);">
        <div style="font-size: 0.8rem; color: #a0aec0;">
            <strong style="color: #fff;">📌 {station_info['region']}</strong><br/>
            🏔️ Elevation: {station_info['elevation']}m<br/>
            🌐 {station_info['lat']:.4f}°N, {station_info['lon']:.4f}°E
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Satellite analysis options
    st.markdown("##### 🛰️ Satellite Analysis")
    analysis_mode = st.radio(
        "Image Source",
        ["🔴 Live Satellite", "📤 Upload Image"],
        index=0,
    )

    uploaded_file = None
    dataset_image_path = None
    fetch_live = False

    if analysis_mode == "📤 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Satellite Image", type=["jpg", "jpeg", "png", "tif"],
        )
    else:
        fetch_live = True
        st.info("🔴 Fetching live satellite imagery for selected station...")

    st.markdown("---")

    # Detection settings
    st.markdown("##### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05)

    st.markdown("---")

    # Model status
    st.markdown("##### 🤖 System Status")
    if detector.model is not None:
        st.success("✅ YOLOv11 Model Active")
        st.caption(f"`{os.path.basename(detector.model_path)}`")
    else:
        st.error("⚠️ Model Not Loaded")
    
    
    st.markdown("---")
    refresh = st.button("🔄 Refresh Live Data", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════
# FETCH LIVE WEATHER DATA
# ════════════════════════════════════════════════════════════════════
weather_data = WeatherDataFetcher.fetch(
    station_info["lat"], station_info["lon"], selected_station
)
weather_risk = WeatherRiskCalculator.calculate(weather_data)
current = weather_data["current"]
history = weather_data["history"]
alert_level = weather_risk["alert_level"]


# ════════════════════════════════════════════════════════════════════
# HEADER BAR
# ════════════════════════════════════════════════════════════════════
data_source = "LIVE" if weather_data.get("source") == "live" else "SIMULATED"
st.markdown(f"""
<div class="header-bar">
    <div class="header-left">
        <h1>🛰️ AI Landslide Monitoring System</h1>
        <div class="tagline">Real-time Prediction & Detection · National Disaster Management Authority</div>
    </div>
    <div class="header-right">
        <div class="live-indicator">
            <span class="live-dot"></span>
            {data_source} DATA
        </div>
        <div class="station-info">
            📍 <strong>{selected_station}</strong> · {station_info['region']} · {station_info['elevation']}m
            <br/>⏱️ {weather_data['fetched_at']}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# ALERT BANNER
# ════════════════════════════════════════════════════════════════════
alert_messages = {
    "SAFE": "✅ ALL CLEAR — No immediate landslide threat detected. Conditions are stable.",
    "WATCH": "👀 WATCH ADVISORY — Environmental conditions are developing. Continue monitoring.",
    "WARNING": "⚠️ WARNING — Elevated landslide risk! Heavy rainfall and saturated soil detected.",
    "DANGER": "🚨 CRITICAL DANGER — EXTREME LANDSLIDE RISK! Immediate evacuation may be required!",
}
alert_css_class = {"SAFE": "safe", "WATCH": "watch", "WARNING": "warning", "DANGER": "danger"}

st.markdown(f"""
<div class="alert-banner alert-{alert_css_class[alert_level]}">
    {alert_messages[alert_level]}
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SECTION 1: LIVE WEATHER CONDITIONS (6 Metric Cards)
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🌦️ Live Weather & Soil Conditions <span class="badge">Real-Time</span></h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

metrics = [
    ("🌡️", f"{current['temperature']}°C", "Temperature", c1),
    ("💧", f"{current['humidity']}%", "Humidity", c2),
    ("🌧️", f"{current['precipitation']} mm", "Rainfall Now", c3),
    ("🌱", f"{current['soil_moisture_avg']}%", "Soil Moisture", c4),
    ("💨", f"{current['wind_speed']} km/h", "Wind Speed", c5),
    ("☁️", f"{current['cloud_cover']}%", "Cloud Cover", c6),
]

for icon, value, label, col in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-icon">{icon}</div>
            <div class="mc-value">{value}</div>
            <div class="mc-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center; color: #718096; margin-top: 0.5rem; font-size: 0.85rem;">
    {current['weather_description']} · 7-day rainfall: <strong style="color: #fff;">{history['rainfall_7d_mm']} mm</strong> ·
    Soil temp: <strong style="color: #fff;">{current['soil_temperature']}°C</strong>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SECTION 2: EXTENDED CLIMATE DATA
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🌬️ Extended Climate Data <span class="badge">Advanced Metrics</span></h2>
</div>
""", unsafe_allow_html=True)

ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)

# Wind direction compass
_wind_deg = current.get('wind_direction', 0) or 0
_compass_dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
_compass = _compass_dirs[int((_wind_deg % 360) / 22.5 + 0.5) % 16]

extended_metrics = [
    ("🧭", f"{_wind_deg}° {_compass}", "Wind Direction", ec1),
    ("💨", f"{current.get('wind_gusts', 0) or 0} km/h", "Wind Gusts", ec2),
    ("🌡️", f"{current.get('dew_point', 0) or 0}°C", "Dew Point", ec3),
    ("🔽", f"{current.get('surface_pressure', 1013) or 1013} hPa", "Pressure", ec4),
    ("👁️", f"{round((current.get('visibility', 10000) or 10000) / 1000, 1)} km", "Visibility", ec5),
    ("☀️", f"{current.get('uv_index', 0) or 0}", "UV Index", ec6),
]

for icon, value, label, col in extended_metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-icon">{icon}</div>
            <div class="mc-value">{value}</div>
            <div class="mc-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

_pressure = current.get('surface_pressure', 1013) or 1013
_max_gust_7d = history.get('max_gust_7d', 0) or 0
pressure_warning = "⚠️ LOW PRESSURE SYSTEM DETECTED" if _pressure < 1005 else ""

st.markdown(f"""
<div style="text-align: center; color: #718096; margin-top: 0.5rem; font-size: 0.85rem;">
    Wind: <strong style="color: #fff;">{current.get('wind_speed', 0)} km/h {_compass}</strong> ·
    Max gust (7d): <strong style="color: #fff;">{_max_gust_7d} km/h</strong> ·
    Pressure: <strong style="color: #fff;">{_pressure} hPa</strong>
    {f'<span style="color: #ff9800; margin-left: 10px;">{pressure_warning}</span>' if pressure_warning else ''}
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SECTION 3: MULTI-FACTOR RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🎯 Multi-Factor Risk Assessment <span class="badge">ML-Powered</span></h2>
</div>
""", unsafe_allow_html=True)

rcol1, rcol2 = st.columns([1, 2])

with rcol1:
    score = weather_risk["overall_score"]
    if score < 30:
        score_color = "#00e676"
        badge_bg = "rgba(0,230,118,0.2)"
    elif score < 50:
        score_color = "#ffc107"
        badge_bg = "rgba(255,193,7,0.2)"
    elif score < 75:
        score_color = "#ff9800"
        badge_bg = "rgba(255,152,0,0.2)"
    else:
        score_color = "#f44336"
        badge_bg = "rgba(244,67,54,0.2)"
    
    st.markdown(f"""
    <div class="risk-score-container">
        <div class="risk-score-label">Combined Risk Score</div>
        <div class="risk-score-value" style="color: {score_color};">{score:.0f}%</div>
        <div class="risk-level-badge" style="background: {badge_bg}; color: {score_color}; border: 1px solid {score_color};">
            {alert_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

with rcol2:
    factors = weather_risk["factors"]
    for name, f in factors.items():
        label = name.replace("_", " ").title()
        fscore = f["score"]
        if fscore < 30:
            bar_color = "#00e676"
        elif fscore < 50:
            bar_color = "#ffc107"
        elif fscore < 75:
            bar_color = "#ff9800"
        else:
            bar_color = "#f44336"
        
        st.markdown(f"""
        <div class="risk-factor">
            <div class="rf-header">
                <span class="rf-name">{label}</span>
                <span class="rf-value"><strong>{f['value']}</strong> {f['unit']} · Score: {fscore}/100</span>
            </div>
            <div class="rf-bar-bg">
                <div class="rf-bar-fill" style="width: {fscore}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SECTION 4: SOIL MOISTURE PROFILE
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🌱 Soil Moisture Profile <span class="badge">Multi-Depth</span></h2>
</div>
""", unsafe_allow_html=True)

sm1, sm2, sm3, sm4 = st.columns(4)
depths = [
    ("Surface (0-1cm)", current["soil_moisture_surface"], sm1),
    ("Shallow (1-3cm)", current["soil_moisture_shallow"], sm2),
    ("Mid (3-9cm)", current["soil_moisture_mid"], sm3),
    ("Deep (9-27cm)", current["soil_moisture_deep"], sm4),
]

for label, val, col in depths:
    if val < 20:
        color = "#00e676"
    elif val < 35:
        color = "#ffc107"
    elif val < 50:
        color = "#ff9800"
    else:
        color = "#f44336"
    
    with col:
        st.markdown(f"""
        <div class="soil-card">
            <div class="soil-depth">{label}</div>
            <div class="soil-value" style="color: {color};">{val}%</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SECTION 5: 7-DAY RAINFALL CHART
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>📊 7-Day Rainfall History</h2>
</div>
""", unsafe_allow_html=True)

if history.get("dates") and history.get("daily_precip"):
    import pandas as pd
    dates = history["dates"]
    precip = history["daily_precip"]
    if len(dates) == len(precip):
        df = pd.DataFrame({"Date": dates, "Rainfall (mm)": precip})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        st.bar_chart(df, color="#00d9ff", height=250)
    else:
        st.info("Rainfall chart data incomplete")
else:
    st.info("No rainfall history available")


# ════════════════════════════════════════════════════════════════════
# SECTION 6: SATELLITE IMAGE ANALYSIS (YOLOv11)
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🛰️ Satellite Image Analysis <span class="badge">YOLOv11</span></h2>
</div>
""", unsafe_allow_html=True)

import cv2

sat_image = None
source_label = "Unknown Source"

if dataset_image_path and os.path.isfile(dataset_image_path):
    sat_image = cv2.imread(dataset_image_path)
    source_label = f"Dataset: {os.path.basename(dataset_image_path)}"
elif uploaded_file is not None:
    from PIL import Image
    pil_img = Image.open(uploaded_file).convert("RGB")
    sat_image = np.array(pil_img)[:, :, ::-1].copy()
    source_label = f"Upload: {uploaded_file.name}"
elif fetch_live:
    with st.spinner(f"🔄 Fetching live satellite imagery for {selected_station}..."):
        sat_image = LiveSatelliteFetcher.fetch_live_image(
            station_info["lat"], station_info["lon"], zoom=14, grid_size=3
        )
    source_label = f"Live Satellite: {selected_station}"

if sat_image is not None:
    processed = preprocessor.prepare_for_inference(sat_image)
    detections = detector.detect(processed, conf_threshold=conf_threshold)

    det_risk_score, det_risk_level = risk_estimator.estimate_risk(
        detections=detections, image=processed
    )

    annotated = visualizer.create_annotated_image(
        processed, detections, det_risk_score, det_risk_level
    )

    # Show images side by side
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**📸 Original Satellite Image**")
        st.image(visualizer.bgr_to_rgb(processed), use_container_width=True)
    with ic2:
        st.markdown("**🔍 YOLOv11 Detection Results**")
        st.image(visualizer.bgr_to_rgb(annotated), use_container_width=True)

    # Detection alerts
    if detections:
        landslide_count = sum(1 for d in detections if d["class_name"] == "landslide")
        debris_count = sum(1 for d in detections if d["class_name"] == "debris_flow")

        if landslide_count > 0:
            st.error(f"🚨 **LANDSLIDE DETECTED** — {landslide_count} landslide zone(s) identified in satellite imagery!")
        if debris_count > 0:
            st.warning(f"⚠️ **DEBRIS FLOW DETECTED** — {debris_count} debris flow channel(s) found!")

        # Detection log
        st.markdown("""
        <div class="section-header">
            <h2>📋 Detection Log</h2>
        </div>
        """, unsafe_allow_html=True)

        for i, det in enumerate(detections):
            cls = det["class_name"]
            conf = det["confidence"]
            bbox = det["bbox"]
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if cls == "landslide":
                card_class = "landslide"
                icon = "🔴"
            elif cls == "debris_flow":
                card_class = "debris"
                icon = "🟠"
            else:
                card_class = "normal"
                icon = "🟢"

            st.markdown(f"""
            <div class="detection-card {card_class}">
                {icon} <strong>{timestamp}</strong> — <strong>{cls.replace('_',' ').title()}</strong>
                &nbsp;|&nbsp; Confidence: <strong>{conf:.1%}</strong>
                &nbsp;|&nbsp; BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]
                &nbsp;|&nbsp; Area: {det['area']:.0f} px²
            </div>
            """, unsafe_allow_html=True)

        # Charts
        try:
            fig = visualizer.create_summary_plot(detections, det_risk_score, det_risk_level)
            st.pyplot(fig)
        except Exception:
            pass

        # Downloads
        dcol1, dcol2, _ = st.columns([1, 1, 2])
        with dcol1:
            import io
            from PIL import Image as PILImage
            ann_rgb = visualizer.bgr_to_rgb(annotated)
            buf = io.BytesIO()
            PILImage.fromarray(ann_rgb).save(buf, format="PNG")
            st.download_button("📥 Download Annotated Image", buf.getvalue(),
                               "landslide_detection.png", "image/png")
        with dcol2:
            results_json = json.dumps({
                "station": selected_station,
                "timestamp": datetime.now().isoformat(),
                "weather": weather_data["current"],
                "weather_risk": weather_risk,
                "detections": detections,
                "detection_risk": {"score": det_risk_score, "level": det_risk_level},
            }, indent=2)
            st.download_button("📥 Download Report (JSON)", results_json,
                               "monitoring_report.json", "application/json")
    else:
        st.success("✅ **TERRAIN STABLE** — No landslide features detected in satellite imagery. Area appears safe.")

else:
    st.info("👆 Select a satellite image source from the sidebar to run YOLOv11 detection analysis.")


# ════════════════════════════════════════════════════════════════════
# SECTION 7: EVENT LOG
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>📜 Monitoring Event Log</h2>
</div>
""", unsafe_allow_html=True)

now_str = datetime.now().strftime("%H:%M:%S")
events = [
    ("🟢", now_str, f"Live weather data fetched for {selected_station}"),
    ("🟢", now_str, f"Risk assessment computed: {alert_level} ({weather_risk['overall_score']:.0f}%)"),
]

if sat_image is not None and detections:
    for d in detections:
        if d["class_name"] == "landslide":
            events.append(("🔴", now_str, f"LANDSLIDE detected (confidence: {d['confidence']:.0%})"))
        elif d["class_name"] == "debris_flow":
            events.append(("🟠", now_str, f"Debris flow detected (confidence: {d['confidence']:.0%})"))
elif sat_image is not None:
    events.append(("🟢", now_str, "Satellite analysis complete: no threats detected"))

events.append(("⚪", now_str, "System monitoring active — next refresh in 5 minutes"))

for icon, time, msg in events:
    st.markdown(f"""
    <div class="event-log-item">
        <span style="font-size: 1.1rem;">{icon}</span>
        <span class="event-time">{time}</span>
        <span>{msg}</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
    <p class="brand">🛰️ AI Landslide Monitoring System v3.0</p>
    <p>Powered by YOLOv11 • Open-Meteo API • PyTorch • Streamlit</p>
    <p>Monitoring {len(MONITORING_STATIONS)} stations across {len(all_regions)} regions</p>
    <p style="margin-top: 12px; font-size: 0.75rem;">© 2024-2025 • National Disaster Management Authority, Government of India</p>
</div>
""", unsafe_allow_html=True)
