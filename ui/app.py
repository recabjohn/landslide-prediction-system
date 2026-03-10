"""
🛰️ AI Landslide Monitoring System
=====================================
Professional real-time monitoring dashboard for landslide
prediction and detection across Indian hill stations.

Features:
  - Live weather data (Open-Meteo API)
  - Real-time soil moisture & humidity monitoring
  - Multi-factor risk prediction with early warnings
  - YOLOv8 satellite imagery analysis
  - Event reporting and alert system

Run:
    streamlit run ui/app.py
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AI Landslide Monitoring System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
/* Top status bar */
.status-bar {
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    padding: 1rem 2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid rgba(100,100,100,0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.status-bar h1 { color: #ffffff; font-size: 1.5rem; font-weight: 800; margin: 0; }
.status-bar .subtitle { color: #cccccc; font-size: 0.85rem; margin-top: 2px; }
.status-bar .live-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; background: #00e676;
    animation: pulse 1.5s infinite;
    margin-right: 6px; vertical-align: middle;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Alert banner */
.alert-banner {
    padding: 0.9rem 1.5rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}
.alert-safe { background: linear-gradient(135deg,#00b09b,#96c93d); color: #fff; }
.alert-watch { background: linear-gradient(135deg,#f7971e,#ffd200); color: #333; }
.alert-warning { background: linear-gradient(135deg,#f85032,#e73827); color: #fff; }
.alert-danger {
    background: linear-gradient(135deg,#d31027,#ea384d); color: #fff;
    box-shadow: 0 0 25px rgba(234,56,77,0.5);
    animation: alert-glow 1.2s infinite alternate;
}
@keyframes alert-glow { 0% { box-shadow: 0 0 15px rgba(234,56,77,0.3); } 100% { box-shadow: 0 0 30px rgba(234,56,77,0.7); } }

/* Metric cards */
.mc {
    background: linear-gradient(145deg, #f0f2f6, #e2e6ea);
    padding: 1rem 1.2rem;
    border-radius: 12px;
    border: 1px solid rgba(100,100,100,0.1);
    text-align: center;
    min-height: 110px;
    color: #111;
}
.mc .mc-icon { font-size: 1.5rem; }
.mc .mc-val {
    font-size: 1.6rem; font-weight: 800; margin: 0.2rem 0;
    background: linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mc .mc-lbl { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

/* Section headers */
.sh {
    color: #333; font-size: 1.2rem; font-weight: 800; padding-bottom: 0.5rem;
    margin-top: 1.5rem; margin-bottom: 1rem;
    border-bottom: 2px solid rgba(102,126,234,0.3);
}

/* Risk factor bars */
.rf-bar { background: #f8f9fa; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; border: 1px solid #e0e0e0; }
.rf-fill { height: 8px; border-radius: 4px; margin-top: 6px; }

/* Detection row */
.det-row {
    background: #f8f9fa; border: 1px solid #eee;
    padding: 0.8rem 1rem; border-radius: 10px;
    margin: 0.4rem 0; border-left: 4px solid; color: #222;
}

/* Sidebar styling tweaks */
[data-testid="stSidebar"] { padding-top: 2rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important; border: none !important;
    padding: 0.6rem 1.5rem !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}

.footer {
    text-align: center; color: #888; font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem; border-top: 1px solid #eee; margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# INITIALIZE
# ================================================================
@st.cache_resource
def init_detector():
    return LandslideDetector(demo_mode=False)

@st.cache_resource
def init_tools():
    return ImagePreprocessor(target_size=640), RiskEstimator(), Visualizer()

detector = init_detector()
preprocessor, risk_estimator, visualizer = init_tools()


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## 🛰️ Monitoring Panel")
    st.markdown("---")

    # Station selector
    st.markdown("### 📍 Monitoring Station")
    station_names = list(MONITORING_STATIONS.keys())
    selected_station = st.selectbox(
        "Select location",
        station_names,
        index=0,
        help="Choose an Indian hill station to monitor"
    )
    station_info = MONITORING_STATIONS[selected_station]
    st.caption(f"📌 {station_info['region']} · {station_info['elevation']}m elevation")
    st.caption(f"🌐 {station_info['lat']:.4f}°N, {station_info['lon']:.4f}°E")

    st.markdown("---")

    # Satellite analysis
    st.markdown("### 🛰️ Satellite Analysis")
    analysis_mode = st.radio(
        "Image source",
        ["Dataset (pre-loaded)", "Upload satellite image", "Live Satellite (Current Station)"],
        index=0,
    )

    uploaded_file = None
    dataset_image_path = None
    fetch_live = False

    if analysis_mode == "Dataset (pre-loaded)":
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        imgs = []
        for split in ["test", "val", "train"]:
            img_dir = os.path.join(dataset_dir, split, "images")
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        imgs.append((f"{split}/{f}", os.path.join(img_dir, f)))
        if imgs:
            sel = st.selectbox("Select image", range(len(imgs)),
                               format_func=lambda i: imgs[i][0])
            dataset_image_path = imgs[sel][1]
    elif analysis_mode == "Upload satellite image":
        uploaded_file = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "tif"],
        )
    else:
        fetch_live = True
        st.info("Will fetch real satellite imagery for the selected monitoring station location.")

    st.markdown("---")

    conf_threshold = st.slider("Detection Threshold", 0.10, 0.95, 0.30, 0.05)

    st.markdown("---")

    # Model status
    st.markdown("### 🤖 Model Status")
    if detector.model is not None:
        st.success("✅ YOLOv8 ACTIVE")
        st.caption(f"`{os.path.basename(detector.model_path)}`")
    else:
        st.error("⚠️ Model not loaded")

    st.markdown("---")
    refresh = st.button("🔄 Refresh Live Data", use_container_width=True, type="primary")


# ================================================================
# FETCH LIVE DATA
# ================================================================
weather_data = WeatherDataFetcher.fetch(
    station_info["lat"], station_info["lon"], selected_station
)
weather_risk = WeatherRiskCalculator.calculate(weather_data)
current = weather_data["current"]
history = weather_data["history"]
alert_level = weather_risk["alert_level"]

# ================================================================
# HEADER + STATUS BAR
# ================================================================
data_badge = "🟢 LIVE" if weather_data.get("source") == "live" else "🟡 SIMULATED"
st.markdown(f"""
<div class="status-bar">
    <div>
        <h1>🛰️ AI Landslide Monitoring System</h1>
        <div class="subtitle">Real-time Prediction & Detection · Indian Terrain Stations</div>
    </div>
    <div style="text-align:right; color:rgba(255,255,255,0.6); font-size:0.85rem;">
        <div><span class="live-dot"></span> {data_badge} · {weather_data['fetched_at']}</div>
        <div style="margin-top:4px;">📍 <strong style="color:#fff;">{selected_station}</strong> · {station_info['region']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# ALERT BANNER
# ================================================================
alert_msgs = {
    "SAFE": "✅ ALL CLEAR — No immediate landslide threat detected for this region",
    "WATCH": "👀 WATCH — Conditions are developing. Monitor closely for changes",
    "WARNING": "⚠️ WARNING — Elevated landslide risk! Heavy rain and saturated soil detected",
    "DANGER": "🚨 DANGER — CRITICAL LANDSLIDE RISK! Immediate action may be required",
}
alert_css = {"SAFE": "safe", "WATCH": "watch", "WARNING": "warning", "DANGER": "danger"}

st.markdown(f"""
<div class="alert-banner alert-{alert_css[alert_level]}">
    {alert_msgs[alert_level]}
</div>
""", unsafe_allow_html=True)


# ================================================================
# SECTION 1: LIVE CONDITIONS (6 metric cards)
# ================================================================
st.markdown('<div class="sh">🌦️ Live Weather & Soil Conditions</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">🌡️</div>
        <div class="mc-val">{current['temperature']}°C</div>
        <div class="mc-lbl">Temperature</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">💧</div>
        <div class="mc-val">{current['humidity']}%</div>
        <div class="mc-lbl">Humidity</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">🌧️</div>
        <div class="mc-val">{current['precipitation']} mm</div>
        <div class="mc-lbl">Rainfall Now</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">🌱</div>
        <div class="mc-val">{current['soil_moisture_avg']}%</div>
        <div class="mc-lbl">Soil Moisture</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">💨</div>
        <div class="mc-val">{current['wind_speed']} km/h</div>
        <div class="mc-lbl">Wind Speed</div>
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""<div class="mc">
        <div class="mc-icon">☁️</div>
        <div class="mc-val">{current['cloud_cover']}%</div>
        <div class="mc-lbl">Cloud Cover</div>
    </div>""", unsafe_allow_html=True)

# Current weather description
st.markdown(f"""
<div style="text-align:center; color:rgba(255,255,255,0.5); margin-top:0.3rem; font-size:0.85rem;">
    {current['weather_description']} · 7-day rainfall: <strong>{history['rainfall_7d_mm']} mm</strong> ·
    Soil temp: <strong>{current['soil_temperature']}°C</strong>
</div>
""", unsafe_allow_html=True)


# ================================================================
# SECTION 2: RISK ASSESSMENT
# ================================================================
st.markdown('<div class="sh">🎯 Multi-Factor Risk Assessment</div>', unsafe_allow_html=True)

rcol1, rcol2 = st.columns([1, 2])

with rcol1:
    score = weather_risk["overall_score"]
    score_color = "#00e676" if score < 30 else "#ffd200" if score < 50 else "#ff5722" if score < 75 else "#ea384d"
    st.markdown(f"""
    <div class="mc" style="min-height: 180px; padding: 1.5rem;">
        <div class="mc-lbl">Combined Risk Score</div>
        <div style="font-size: 3rem; font-weight: 900; color: {score_color}; margin: 0.5rem 0;">{score:.0f}%</div>
        <div class="alert-banner alert-{alert_css[alert_level]}" style="margin: 0; padding: 0.4rem 1rem; font-size: 0.85rem;">
            {alert_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

with rcol2:
    factors = weather_risk["factors"]
    for name, f in factors.items():
        label = name.replace("_", " ").title()
        bar_color = "#00e676" if f["score"] < 30 else "#ffd200" if f["score"] < 50 else "#ff5722" if f["score"] < 75 else "#ea384d"
        st.markdown(f"""
        <div class="rf-bar">
            <div style="display:flex; justify-content:space-between; color:#333; font-size:0.9rem;">
                <span>{label}</span>
                <span><strong>{f['value']}</strong> {f['unit']} · Score: {f['score']}/100</span>
            </div>
            <div style="background:#e0e0e0; border-radius:4px; height:8px; margin-top:8px;">
                <div class="rf-fill" style="width:{f['score']}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# SECTION 3: SOIL MOISTURE BREAKDOWN
# ================================================================
st.markdown('<div class="sh">🌱 Soil Moisture Profile</div>', unsafe_allow_html=True)

sm1, sm2, sm3, sm4 = st.columns(4)
depths = [
    ("Surface (0-1cm)", current["soil_moisture_surface"]),
    ("Shallow (1-3cm)", current["soil_moisture_shallow"]),
    ("Mid (3-9cm)", current["soil_moisture_mid"]),
    ("Deep (9-27cm)", current["soil_moisture_deep"]),
]
for col, (label, val) in zip([sm1, sm2, sm3, sm4], depths):
    bar_color = "#00e676" if val < 20 else "#ffd200" if val < 35 else "#ff5722" if val < 50 else "#ea384d"
    with col:
        st.markdown(f"""<div class="mc" style="min-height: 90px;">
            <div class="mc-lbl">{label}</div>
            <div style="font-size:1.4rem; font-weight:800; color:{bar_color}; margin:0.2rem 0;">{val}%</div>
        </div>""", unsafe_allow_html=True)


# ================================================================
# SECTION 4: 7-DAY RAINFALL CHART
# ================================================================
st.markdown('<div class="sh">📊 7-Day Rainfall History</div>', unsafe_allow_html=True)

if history.get("dates") and history.get("daily_precip"):
    import pandas as pd
    dates = history["dates"]
    precip = history["daily_precip"]
    if len(dates) == len(precip):
        df = pd.DataFrame({"Date": dates, "Rainfall (mm)": precip})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        st.bar_chart(df, color="#667eea", height=200)
    else:
        st.info("Rainfall chart data incomplete")
else:
    st.info("No rainfall history available")


# ================================================================
# SECTION 5: SATELLITE ANALYSIS (YOLO)
# ================================================================
st.markdown('<div class="sh">🛰️ Satellite Image Analysis (YOLOv8)</div>', unsafe_allow_html=True)

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
    with st.spinner(f"Fetching live satellite tile for {selected_station}..."):
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
        st.markdown("**📸 Satellite Image**")
        st.image(visualizer.bgr_to_rgb(processed), use_column_width=True)
    with ic2:
        st.markdown("**🔍 Detection Results**")
        st.image(visualizer.bgr_to_rgb(annotated), use_column_width=True)

    # Detection details
    if detections:
        landslide_count = sum(1 for d in detections if d["class_name"] == "landslide")
        debris_count = sum(1 for d in detections if d["class_name"] == "debris_flow")

        if landslide_count > 0:
            st.error(f"🚨 **LANDSLIDE DETECTED** — {landslide_count} landslide zone(s) identified in satellite imagery!")
        if debris_count > 0:
            st.warning(f"⚠️ **DEBRIS FLOW DETECTED** — {debris_count} debris flow channel(s) found!")

        # Detection table
        st.markdown('<div class="sh">📋 Detection Log</div>', unsafe_allow_html=True)

        class_colors = {"landslide": "#ff4444", "debris_flow": "#ff9800", "normal_terrain": "#4caf50"}
        for i, det in enumerate(detections):
            cls = det["class_name"]
            conf = det["confidence"]
            bbox = det["bbox"]
            color = class_colors.get(cls, "#888")
            timestamp = datetime.now().strftime("%H:%M:%S")

            st.markdown(f"""
            <div class="det-row" style="border-left-color: {color};">
                <strong style="color:{color};">⏱ {timestamp} — {cls.replace('_',' ').title()}</strong>
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
        st.success("✅ No landslide features detected in the satellite image. Terrain appears stable.")

else:
    st.info("👆 Select a satellite image from the sidebar to run YOLO detection analysis.")


# ================================================================
# SECTION 6: EVENT LOG
# ================================================================
st.markdown('<div class="sh">📜 Monitoring Event Log</div>', unsafe_allow_html=True)

now_str = datetime.now().strftime("%H:%M:%S")
events = [
    f"🟢 {now_str} — Live weather data fetched for {selected_station}",
    f"🟢 {now_str} — Risk assessment computed: {alert_level} ({weather_risk['overall_score']:.0f}%)",
]
if sat_image is not None and detections:
    for d in detections:
        if d["class_name"] == "landslide":
            events.append(f"🔴 {now_str} — LANDSLIDE detected (conf: {d['confidence']:.0%})")
        elif d["class_name"] == "debris_flow":
            events.append(f"🟠 {now_str} — Debris flow detected (conf: {d['confidence']:.0%})")
elif sat_image is not None:
    events.append(f"🟢 {now_str} — Satellite analysis complete: no threats detected")

events.append(f"⚪ {now_str} — System monitoring active")

for event in events:
    st.markdown(f"<div style='color:#555; font-size:0.9rem; padding:0.3rem 0; border-bottom:1px solid #eee;'>{event}</div>",
                unsafe_allow_html=True)


# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div class="footer">
    <p>🛰️ AI Landslide Monitoring System v2.0 · Indian Terrain Stations</p>
    <p>YOLOv8 • Open-Meteo API • PyTorch • OpenCV • Streamlit</p>
    <p>Final Year Engineering Project • 2024-2025</p>
</div>
""", unsafe_allow_html=True)
