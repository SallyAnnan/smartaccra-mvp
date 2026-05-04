import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SmartAccra",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# FRIEND'S DESIGN SYSTEM INJECTED AS CSS
# All colours, fonts, cards from the HTML files
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Public+Sans:wght@400;700&display=swap');

:root {
    --primary: #001b0f;
    --primary-container: #013220;
    --secondary: #795900;
    --secondary-container: #ffbf00;
    --background: #f9f9f9;
    --on-surface: #1a1c1c;
    --on-surface-variant: #414943;
    --error: #ba1a1a;
    --surface-tint: #3b6751;
}

.main { background-color: #f9f9f9 !important; }
.stApp { background-color: #f9f9f9 !important; }
* { font-family: 'Public Sans', sans-serif !important; }
h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #001b0f !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* RISK CARDS — from friend's home dashboard */
.risk-card {
    background: white; border-radius: 24px; padding: 20px; margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #f8fafc;
    display: flex; align-items: center; gap: 16px;
}
.risk-icon-wrap {
    width: 56px; height: 56px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; flex-shrink: 0;
}
.risk-icon-flood { background: #eff6ff; }
.risk-icon-fire { background: #fff7ed; }
.risk-icon-waste { background: #f0fdf4; }

/* RISK BADGES */
.risk-badge { padding: 4px 12px; border-radius: 999px; font-size: 10px; font-weight: 700; letter-spacing: 0.05em; }
.badge-high { background: #ffdad6; color: #ba1a1a; }
.badge-moderate { background: #fff3cc; color: #795900; }
.badge-low { background: #d1fae5; color: #3b6751; }
.badge-collection { background: #d1fae5; color: #3b6751; font-size: 9px; }

/* ORIGINAL RISK SPANS — kept for backward compat */
.risk-high { background: #ba1a1a; color: white; padding: 8px 18px; border-radius: 20px; font-weight: 700; display: inline-block; font-size: 0.95em; }
.risk-moderate { background: #ffbf00; color: #261a00; padding: 8px 18px; border-radius: 20px; font-weight: 700; display: inline-block; font-size: 0.95em; }
.risk-low { background: #34c759; color: white; padding: 8px 18px; border-radius: 20px; font-weight: 700; display: inline-block; font-size: 0.95em; }

/* ALERT BANNER — friend's style */
.alert-banner {
    background: #ba1a1a; color: white; padding: 16px 20px; border-radius: 16px; margin: 8px 0;
    display: flex; align-items: center; gap: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600; font-size: 14px;
    box-shadow: 0 4px 12px rgba(186,26,26,0.2);
}

/* INFO CARDS — friend's action card style */
.info-card {
    background: white; padding: 16px 20px; border-radius: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin: 8px 0;
    border: 1px solid #f1f5f9; display: flex; align-items: flex-start; gap: 16px;
}
.info-icon {
    width: 48px; height: 48px; border-radius: 16px; background: #001b0f;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 22px; flex-shrink: 0;
}
.info-card h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 600; font-size: 15px; color: #001b0f; margin: 0 0 4px 0; }
.info-card p { font-size: 13px; color: #414943; margin: 0; line-height: 1.5; }

/* METRIC CARD */
.metric-card {
    background: white; padding: 24px; border-radius: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03); text-align: center; margin: 8px 0; border: 1px solid #f8fafc;
}

/* WELCOME CARD */
.welcome-card {
    background: linear-gradient(to bottom, rgba(1,50,32,0.85), rgba(0,27,15,0.97));
    color: white; padding: 48px 32px; border-radius: 24px; margin: 20px 0;
    text-align: center; box-shadow: 0 8px 32px rgba(0,27,15,0.3);
}

/* STREAMLIT BUTTON */
.stButton button {
    background: #ffbf00 !important; color: #261a00 !important;
    border: none !important; padding: 14px 24px !important; border-radius: 14px !important;
    font-weight: 700 !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important; width: 100% !important;
    box-shadow: 0 4px 16px rgba(255,191,0,0.3) !important;
}
.stButton button:hover { background: #e6b000 !important; color: #261a00 !important; }

/* HAZARD FEED */
.hazard-item {
    background: white; border-radius: 24px; padding: 20px; margin: 8px 0;
    border: 1px solid rgba(193,200,194,0.2); box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.hazard-header { display: flex; align-items: flex-start; gap: 16px; }
.hazard-icon {
    width: 48px; height: 48px; border-radius: 12px; background: #fff3e0;
    display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0;
}
.hazard-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; }
.confirm-btn {
    background: #001b0f; color: white; border: none; padding: 6px 16px; border-radius: 8px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.05em; cursor: pointer;
}

/* FIRE RISK FACTOR ROWS */
.factor-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 16px; background: #f3f3f4; border-radius: 12px; margin: 6px 0;
    border: 1px solid rgba(193,200,194,0.3);
}
.factor-label { display: flex; align-items: center; gap: 10px; font-size: 15px; font-weight: 600; color: #001b0f; }
.factor-value { font-size: 15px; font-weight: 600; }
.factor-value.high { color: #ba1a1a; }
.factor-value.normal { color: #1a1c1c; }

/* STAT GRID */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 16px 0; }
.stat-tile {
    background: white; border-radius: 16px; padding: 14px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #f1f5f9;
}
.stat-val { font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 22px; font-weight: 700; }
.stat-label { font-size: 10px; font-weight: 700; letter-spacing: 0.05em; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }
.stat-val.red { color: #ba1a1a; }
.stat-val.green { color: #001b0f; }
.stat-val.amber { color: #795900; }

/* ALERTS FEED */
.alert-item {
    background: white; border-radius: 24px; padding: 20px; margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); display: flex; gap: 16px; align-items: flex-start;
}
.alert-item.flood { border-left: 6px solid #3b82f6; }
.alert-item.fire { border-left: 6px solid #ef4444; }
.alert-item.waste { border-left: 6px solid #f59e0b; }
.alert-item.resolved { border-left: 6px solid #10b981; opacity: 0.8; }
.alert-icon { padding: 12px; border-radius: 16px; font-size: 22px; flex-shrink: 0; }
.alert-icon.flood { background: #eff6ff; }
.alert-icon.fire { background: #fef2f2; }
.alert-icon.waste { background: #fffbeb; }
.alert-icon.resolved { background: #f0fdf4; }

/* SECTION HEADER */
.section-header {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 20px; font-weight: 700; color: #001b0f; margin: 24px 0 12px 0;
}

/* CALENDAR STRIP */
.calendar-strip { display: flex; gap: 10px; overflow-x: auto; padding: 8px 0; scrollbar-width: none; }
.calendar-strip::-webkit-scrollbar { display: none; }
.cal-day {
    min-width: 60px; height: 76px; border-radius: 16px; background: #eeeeee;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 2px; color: #1a1c1c; flex-shrink: 0;
}
.cal-day .day-name { font-size: 11px; font-weight: 700; letter-spacing: 0.05em; opacity: 0.6; }
.cal-day .day-num { font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 18px; font-weight: 700; }
.cal-day.today { background: #ffbf00; color: #261a00; transform: scale(1.05); }
.cal-day.today .day-name { opacity: 1; }
.cal-day.collection { background: #013220; color: white; }

/* METRIC STRIP */
.metric-strip { display: flex; gap: 12px; overflow-x: auto; padding: 4px 0 12px 0; scrollbar-width: none; }
.metric-strip::-webkit-scrollbar { display: none; }
.metric-tile {
    min-width: 130px; background: white; border-radius: 24px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #f1f5f9; flex-shrink: 0;
}
.metric-tile .label { font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: #414943; text-transform: uppercase; margin-bottom: 6px; }
.metric-tile .value { font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 22px; font-weight: 700; color: #052e16; }
.metric-tile-warning { background: rgba(255,191,0,0.1); border: 1px solid rgba(255,191,0,0.2); }
.metric-tile-warning .label { color: #795900; }
.metric-tile-warning .value { color: #795900; }

/* GREETING */
.greeting-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }
.greeting-row h2 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 24px; font-weight: 700; color: #001b0f; margin: 0 0 8px 0; }
.location-chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px;
    background: #f3f3f4; border-radius: 12px; border: 1px solid rgba(193,200,194,0.3);
    font-size: 13px; color: #1a1c1c; cursor: pointer;
}
.avatar {
    width: 48px; height: 48px; border-radius: 999px;
    background: linear-gradient(135deg, #2d6a4f, #001b0f);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 18px; font-weight: 700; flex-shrink: 0;
}

/* FORECAST CARD */
.forecast-card {
    background: white; border-radius: 28px; padding: 24px; margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #f1f5f9;
}
.forecast-card h2 { font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 20px; font-weight: 700; color: #001b0f; margin: 0 0 4px 0; }
.forecast-card p { font-size: 13px; color: #414943; margin: 0 0 16px 0; }

/* HAZARD TYPE GRID */
.hazard-type-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0; }
.hazard-type-btn {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 20px 12px; border: 2px solid #e2e2e2; border-radius: 24px; background: white; cursor: pointer; gap: 8px;
}
.hazard-type-btn.selected { border: 2px solid #ffbf00; box-shadow: 0 0 0 3px rgba(255,191,0,0.15); }

/* ACTION BUTTONS */
.primary-action-btn {
    width: 100%; padding: 20px; background: #ffbf00; color: #261a00; border: none;
    border-radius: 16px; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 16px; font-weight: 700; cursor: pointer; display: flex;
    align-items: center; justify-content: center; gap: 10px;
    box-shadow: 0 4px 16px rgba(255,191,0,0.3); margin-top: 16px;
}
.danger-action-btn {
    width: 100%; padding: 20px; background: #ba1a1a; color: white; border: none;
    border-radius: 16px; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 16px; font-weight: 700; cursor: pointer; display: flex;
    align-items: center; justify-content: center; gap: 10px;
    box-shadow: 0 4px 16px rgba(186,26,26,0.2); margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACCRA DISTRICTS — Real coordinates + risk profiles
# ─────────────────────────────────────────────
ACCRA_DISTRICTS = {
    # HIGH RISK — Slums / Informal Settlements
    "Nima": {"lat": 5.5833, "lon": -0.2056, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Chorkor": {"lat": 5.5333, "lon": -0.2333, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Old Fadama": {"lat": 5.5500, "lon": -0.2167, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Agbogbloshie": {"lat": 5.5481, "lon": -0.2233, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "James Town": {"lat": 5.5333, "lon": -0.2167, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Sukura": {"lat": 5.5450, "lon": -0.2350, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Sabon Zongo": {"lat": 5.5550, "lon": -0.2300, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Mamobi": {"lat": 5.5867, "lon": -0.2100, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Ashaiman": {"lat": 5.7000, "lon": -0.0333, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "HIGH", "waste_risk": "HIGH"},
    "Mamprobi": {"lat": 5.5333, "lon": -0.2333, "category": "Informal Settlement", "flood_risk": "HIGH", "fire_risk": "MODERATE", "waste_risk": "HIGH"},
    # MODERATE-HIGH RISK — Dense Mixed Areas
    "Kaneshie": {"lat": 5.5667, "lon": -0.2333, "category": "Dense Mixed", "flood_risk": "HIGH", "fire_risk": "MODERATE", "waste_risk": "HIGH"},
    "Adabraka": {"lat": 5.5583, "lon": -0.2056, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Lapaz": {"lat": 5.6042, "lon": -0.2347, "category": "Dense Mixed", "flood_risk": "HIGH", "fire_risk": "MODERATE", "waste_risk": "HIGH"},
    "Darkuman": {"lat": 5.5833, "lon": -0.2500, "category": "Dense Mixed", "flood_risk": "HIGH", "fire_risk": "MODERATE", "waste_risk": "HIGH"},
    "Bubuashie": {"lat": 5.5667, "lon": -0.2417, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Odorkor": {"lat": 5.5750, "lon": -0.2667, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Abeka": {"lat": 5.6000, "lon": -0.2333, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Kotobabi": {"lat": 5.5917, "lon": -0.2167, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Alajo": {"lat": 5.6000, "lon": -0.2167, "category": "Dense Mixed", "flood_risk": "HIGH", "fire_risk": "MODERATE", "waste_risk": "HIGH"},
    "Pig Farm": {"lat": 5.5917, "lon": -0.2083, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Kwashieman": {"lat": 5.5833, "lon": -0.2750, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    "Sakaman": {"lat": 5.5667, "lon": -0.2583, "category": "Dense Mixed", "flood_risk": "MODERATE", "fire_risk": "MODERATE", "waste_risk": "MODERATE"},
    # MODERATE RISK — Working Class Suburbs
    "Adenta": {"lat": 5.7167, "lon": -0.1667, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Madina": {"lat": 5.6833, "lon": -0.1667, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Medina": {"lat": 5.6667, "lon": -0.2000, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Dansoman": {"lat": 5.5500, "lon": -0.2667, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Tema": {"lat": 5.6667, "lon": -0.0167, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Kasoa": {"lat": 5.5333, "lon": -0.4167, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Achimota": {"lat": 5.6333, "lon": -0.2333, "category": "Working Class Suburb", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Ablekuma": {"lat": 5.5833, "lon": -0.2917, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Taifa": {"lat": 5.6667, "lon": -0.2333, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Dome": {"lat": 5.6500, "lon": -0.2167, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Haatso": {"lat": 5.6500, "lon": -0.2000, "category": "Working Class Suburb", "flood_risk": "MODERATE", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    "Kwabenya": {"lat": 5.7000, "lon": -0.2167, "category": "Working Class Suburb", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "MODERATE"},
    # LOW RISK — Affluent / Planned Areas
    "Airport Residential": {"lat": 5.6042, "lon": -0.1733, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "East Legon": {"lat": 5.6500, "lon": -0.1500, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Cantonments": {"lat": 5.5833, "lon": -0.1667, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Labone": {"lat": 5.5667, "lon": -0.1750, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Osu": {"lat": 5.5556, "lon": -0.1828, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Ridge": {"lat": 5.5667, "lon": -0.2000, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Roman Ridge": {"lat": 5.5917, "lon": -0.1917, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Dzorwulu": {"lat": 5.6083, "lon": -0.1917, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
    "Abelemkpe": {"lat": 5.6000, "lon": -0.2000, "category": "Affluent Planned", "flood_risk": "LOW", "fire_risk": "LOW", "waste_risk": "LOW"},
}

RISK_SCORE = {"HIGH": 85, "MODERATE": 55, "LOW": 25}
RISK_COLOR = {"HIGH": "#ff4444", "MODERATE": "#ff9500", "LOW": "#34c759"}
RISK_FOLIUM = {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}

# ─────────────────────────────────────────────
# WEATHER API — Open-Meteo (real data)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=precipitation,temperature_2m"
        f"&daily=precipitation_sum,temperature_2m_max,precipitation_probability_max"
        f"&timezone=Africa%2FAccra&forecast_days=7"
    )
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception:
        return None

# ─────────────────────────────────────────────
# DYNAMIC FLOOD RISK
# ─────────────────────────────────────────────
def calc_dynamic_flood(district, rainfall_mm):
    base = ACCRA_DISTRICTS[district]["flood_risk"]
    base_score = RISK_SCORE[base]
    rain_boost = min(rainfall_mm * 1.5, 30)
    final = min(base_score + rain_boost, 100)
    if final >= 70: return "HIGH", final
    if final >= 40: return "MODERATE", final
    return "LOW", final

# ─────────────────────────────────────────────
# FIRE RISK MODEL — Logistic Regression
# ─────────────────────────────────────────────
@st.cache_resource
def train_fire_model():
    np.random.seed(42)
    n = 600
    density = np.random.uniform(0.1, 1.0, n)
    wiring_age = np.random.uniform(1, 30, n)
    temp = np.random.uniform(25, 38, n)
    dry = np.random.randint(0, 2, n)
    reports = np.random.randint(0, 25, n)
    score = (0.35*density + 0.25*(wiring_age/30) + 0.15*((temp-25)/13) + 0.15*dry + 0.10*(reports/25))
    y = (score > 0.5).astype(int)
    X = np.column_stack([density, wiring_age, temp, dry, reports])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = LogisticRegression(random_state=42)
    m.fit(Xs, y)
    return m, sc

def predict_fire(district, temp=32, dry_season=1, reports=5):
    m, sc = train_fire_model()
    cat = ACCRA_DISTRICTS[district]["category"]
    density_map = {"Informal Settlement": 0.95, "Dense Mixed": 0.75, "Working Class Suburb": 0.50, "Affluent Planned": 0.25}
    density = density_map.get(cat, 0.5)
    wiring_age = 22 if cat == "Informal Settlement" else 15 if cat == "Dense Mixed" else 8
    X = np.array([[density, wiring_age, temp, dry_season, reports]])
    Xs = sc.transform(X)
    p = m.predict_proba(Xs)[0][1]
    if p >= 0.65: return "HIGH", p*100
    if p >= 0.35: return "MODERATE", p*100
    return "LOW", p*100

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_district" not in st.session_state:
    st.session_state.user_district = "Adenta"

# ─────────────────────────────────────────────
# ONBOARDING — friend's welcome screen style
# ─────────────────────────────────────────────
if not st.session_state.onboarded:
    st.markdown("""
    <div class='welcome-card'>
        <div style='font-size:64px;margin-bottom:16px'>🛡️</div>
        <h1 style='color:white;margin:0;font-family:Plus Jakarta Sans,sans-serif;font-size:2.6em;font-weight:800;letter-spacing:-1px'>SmartAccra</h1>
        <p style='font-size:1.2em;margin:12px 0 8px;opacity:0.9;font-weight:600'>Know before it happens.</p>
        <p style='opacity:0.7;margin:0;font-size:0.95em'>Urban Risk Intelligence for Accra, Ghana</p>
        <div style='display:flex;justify-content:center;gap:20px;margin-top:28px;flex-wrap:wrap'>
            <div style='background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:16px;padding:16px 20px;text-align:center'>
                <div style='font-size:28px;margin-bottom:6px'>🌊</div>
                <div style='font-size:12px;font-weight:600;opacity:0.9'>Flood Alerts</div>
            </div>
            <div style='background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:16px;padding:16px 20px;text-align:center'>
                <div style='font-size:28px;margin-bottom:6px'>🗺️</div>
                <div style='font-size:12px;font-weight:600;opacity:0.9'>Risk Maps</div>
            </div>
            <div style='background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:16px;padding:16px 20px;text-align:center'>
                <div style='font-size:28px;margin-bottom:6px'>🔥</div>
                <div style='font-size:12px;font-weight:600;opacity:0.9'>Fire Risk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👋 Let's personalise your experience")
    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input("👤 Your first name", placeholder="e.g. Kwame")
    with col2:
        district = st.selectbox("📍 Your neighbourhood", sorted(ACCRA_DISTRICTS.keys()))

    st.markdown("Or type a specific address (optional):")
    custom_address = st.text_input("🏠 Street address or landmark", placeholder="e.g. Near Medina Market")

    if st.button("🚀 Continue to SmartAccra", use_container_width=True):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.user_district = district
            st.session_state.user_address = custom_address
            st.session_state.onboarded = True
            st.rerun()
        else:
            st.warning("Please enter your name to continue")

    st.caption("🔒 Sign in with Google coming in Phase 2 • Your data stays on your device")
    st.stop()

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
user_name = st.session_state.user_name
selected = st.session_state.user_district
info = ACCRA_DISTRICTS[selected]
lat, lon = info["lat"], info["lon"]

weather = fetch_weather(lat, lon)
if weather and "daily" in weather:
    d = weather["daily"]
    rain_today = d["precipitation_sum"][0] or 0
    rain_tomorrow = d["precipitation_sum"][1] if len(d["precipitation_sum"]) > 1 else 0
    rain_prob = d["precipitation_probability_max"][0] or 0
else:
    rain_today, rain_tomorrow, rain_prob = 5, 15, 60

flood_lvl, flood_pct = calc_dynamic_flood(selected, rain_tomorrow)
fire_lvl, fire_pct = predict_fire(selected)
waste_lvl = info["waste_risk"]

# ─────────────────────────────────────────────
# SIDEBAR — original functionality intact
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 Hi, {user_name}!")
    st.markdown(f"📍 **{selected}**")
    st.caption(f"Category: {info['category']}")
    st.markdown("---")

    new_district = st.selectbox(
        "Change district",
        sorted(ACCRA_DISTRICTS.keys()),
        index=sorted(ACCRA_DISTRICTS.keys()).index(selected)
    )
    if new_district != selected:
        st.session_state.user_district = new_district
        st.rerun()

    st.markdown("---")
    page = st.radio(
        "🧭 Navigation",
        ["🏠 Home", "🌊 Flood Risk", "🔥 Fire Risk", "🗑️ Waste & Drainage", "📢 Report Hazard", "🔔 Alerts", "ℹ️ About"]
    )

    st.markdown("---")
    if st.button("🚪 Sign out"):
        st.session_state.onboarded = False
        st.rerun()
    st.caption("SmartAccra v1.0 • SDG 11")

# ─────────────────────────────────────────────
# HOME — friend's card design + original data
# ─────────────────────────────────────────────
if page == "🏠 Home":
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    initial = user_name[0].upper() if user_name else "K"

    st.markdown(f"""
    <div class="greeting-row">
        <div>
            <h2>{greet}, {user_name}!</h2>
            <div class="location-chip">📍 {selected} <span style="opacity:0.5">▾</span></div>
        </div>
        <div class="avatar">{initial}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"*{datetime.now().strftime('%A, %d %B %Y')}*")

    if flood_lvl == "HIGH" or fire_lvl == "HIGH":
        st.markdown(f"""
        <div class="alert-banner">
            <span style="font-size:22px">⚠️</span>
            <span>Active alert: High risk detected in {selected}. See details below.</span>
            <span style="margin-left:auto;font-size:20px">›</span>
        </div>
        """, unsafe_allow_html=True)

    def badge_class(lvl):
        return "badge-high" if lvl == "HIGH" else "badge-moderate" if lvl == "MODERATE" else "badge-low"

    st.markdown(f"""
    <div class="risk-card">
        <div class="risk-icon-wrap risk-icon-flood">💧</div>
        <div style="flex:1">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:16px;color:#001b0f">Flood Risk</span>
                <span class="risk-badge {badge_class(flood_lvl)}">{flood_lvl}</span>
            </div>
            <p style="font-size:13px;color:#414943;margin:0">Rain probability {int(rain_prob)}% · {rain_tomorrow:.1f}mm forecast tomorrow</p>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-icon-wrap risk-icon-fire">🔥</div>
        <div style="flex:1">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:16px;color:#001b0f">Fire Risk</span>
                <span class="risk-badge {badge_class(fire_lvl)}">{fire_lvl}</span>
            </div>
            <p style="font-size:13px;color:#414943;margin:0">{info['category']} · Risk score {fire_pct:.0f}%</p>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-icon-wrap risk-icon-waste">🗑️</div>
        <div style="flex:1">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:16px;color:#001b0f">Waste & Drainage</span>
                <span class="risk-badge badge-collection">COLLECTION TOMORROW</span>
            </div>
            <p style="font-size:13px;color:#414943;margin:0">{'3 blockage reports near you' if waste_lvl == 'HIGH' else '1 report nearby' if waste_lvl == 'MODERATE' else 'All clear'}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">🗺️ Accra Risk Overview</p>', unsafe_allow_html=True)
    m = folium.Map(location=[5.6037, -0.1870], zoom_start=11, tiles="OpenStreetMap")
    for d, dinfo in ACCRA_DISTRICTS.items():
        scores = [RISK_SCORE[dinfo["flood_risk"]], RISK_SCORE[dinfo["fire_risk"]], RISK_SCORE[dinfo["waste_risk"]]]
        avg = sum(scores) / 3
        color = "red" if avg >= 70 else "orange" if avg >= 50 else "beige" if avg >= 35 else "green"
        folium.CircleMarker(
            location=[dinfo["lat"], dinfo["lon"]], radius=10,
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{d}</b><br>Type: {dinfo['category']}<br>Flood: {dinfo['flood_risk']}<br>Fire: {dinfo['fire_risk']}<br>Waste: {dinfo['waste_risk']}", max_width=220),
            tooltip=d
        ).add_to(m)
    folium.Marker(
        location=[lat, lon], popup=f"You: {selected}",
        tooltip=f"📍 {selected}",
        icon=folium.Icon(color="darkgreen", icon="home")
    ).add_to(m)
    st_folium(m, width=None, height=480, returned_objects=[])
    st.caption("🔴 High Risk  🟠 Moderate-High  🟡 Moderate  🟢 Low Risk")

# ─────────────────────────────────────────────
# FLOOD RISK — friend's metric strip + all original charts
# ─────────────────────────────────────────────
elif page == "🌊 Flood Risk":
    st.markdown(f"## 🌊 Flood Risk — {selected}")
    st.caption("Real-time analysis using Open-Meteo weather API + neighbourhood vulnerability profiling")

    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-tile">
            <div style="font-size:24px;margin-bottom:6px">🌧️</div>
            <div class="label">Today</div>
            <div class="value">{rain_today:.0f}mm</div>
        </div>
        <div class="metric-tile metric-tile-warning">
            <div style="font-size:24px;margin-bottom:6px">⚠️</div>
            <div class="label">Risk Level</div>
            <div class="value">{flood_lvl.title()}</div>
        </div>
        <div class="metric-tile">
            <div style="font-size:24px;margin-bottom:6px">🌂</div>
            <div class="label">Tomorrow</div>
            <div class="value">{rain_tomorrow:.0f}mm</div>
        </div>
        <div class="metric-tile">
            <div style="font-size:24px;margin-bottom:6px">📊</div>
            <div class="label">Probability</div>
            <div class="value">{int(rain_prob)}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card" style="text-align:center;margin:20px 0">
        <h3 style="margin:0 0 12px 0">Current Flood Risk</h3>
        <span class="risk-{flood_lvl.lower()}" style="font-size:1.3em">{flood_lvl}</span>
        <p style="color:#666;margin-top:14px">Risk score: {flood_pct:.1f} / 100</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="forecast-card">
        <h2>Flood Forecast</h2>
        <p>7-day rainfall and risk projection for your area</p>
    </div>
    """, unsafe_allow_html=True)

    if weather and "daily" in weather:
        d = weather["daily"]
        dates = d["time"][:7]
        precip = [p or 0 for p in d["precipitation_sum"][:7]]
        prob = [p or 0 for p in d["precipitation_probability_max"][:7]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=precip, name="Rainfall (mm)",
            marker_color=["#ba1a1a" if p > 20 else "#ffbf00" if p > 10 else "#34c759" for p in precip]))
        fig.add_trace(go.Scatter(x=dates, y=prob, name="Probability (%)", yaxis="y2",
            line=dict(color="#3b6751", width=2.5), mode="lines+markers"))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Rainfall (mm)",
            yaxis2=dict(title="Probability (%)", overlaying="y", side="right", range=[0, 100]),
            plot_bgcolor="white", paper_bgcolor="white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 🗺️ Flood Map — {selected}")
    fm = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker(location=[lat, lon], radius=400,
        color=RISK_FOLIUM[flood_lvl], fill=True, fill_opacity=0.25,
        popup=f"{selected} — {flood_lvl}", tooltip=f"Flood Zone: {flood_lvl}").add_to(fm)
    folium.Marker(location=[lat, lon], popup=selected,
        icon=folium.Icon(color=RISK_FOLIUM[flood_lvl], icon="tint")).add_to(fm)
    st_folium(fm, width=None, height=380, returned_objects=[])

    st.markdown("### ⚡ What should you do?")
    if flood_lvl == "HIGH":
        actions = [("📦", "Move valuables", "Relocate electronics and documents to upper levels or high shelves immediately."),
                   ("🧹", "Clear entrance drain", "Ensure the gutter in front of your property is free of waste and debris."),
                   ("📢", "Warn neighbours", "Check on elderly residents and share updates via community groups."),
                   ("🏥", "Know your route", "Identify your nearest high ground and evacuation route."),
                   ("💧", "Store clean water", "Keep emergency supplies ready for 48 hours.")]
    elif flood_lvl == "MODERATE":
        actions = [("🧹", "Clear gutters", "Remove debris from gutters and drains near your home."),
                   ("📦", "Raise documents", "Move important papers to higher shelves as a precaution."),
                   ("👀", "Monitor updates", "Check SmartAccra over the next 24 hours."),
                   ("📱", "Warn others", "Alert neighbours in low-lying areas.")]
    else:
        actions = [("✅", "All clear", "No immediate action needed. Conditions are safe."),
                   ("🧹", "Preventive cleaning", "Good time to clear gutters before rainy season."),
                   ("📱", "Stay informed", "Keep SmartAccra notifications enabled.")]

    for icon, title, desc in actions:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">{icon}</div>
            <div><h3>{title}</h3><p>{desc}</p></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<button class="primary-action-btn">🚨 Report a blocked gutter near me</button>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FIRE RISK — friend's design + original ML + all original features
# ─────────────────────────────────────────────
elif page == "🔥 Fire Risk":
    st.markdown(f"## 🔥 Fire Risk — {selected}")
    st.caption("Predicted using a Logistic Regression model trained on building density, electrical infrastructure age, temperature, seasonal patterns, and community reports.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Area type", info["category"].replace(" Suburb","").replace(" Settlement",""))
    col2.metric("Risk score", f"{fire_pct:.0f}%")
    col3.metric("Fire service ETA", "~45 min" if info["category"] == "Informal Settlement" else "~25 min" if info["category"] == "Dense Mixed" else "~15 min")

    st.markdown(f"""
    <div style="background:white;border-radius:24px;padding:24px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,0.04);border:1px solid #f1f5f9">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px">
            <div>
                <h2 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:700;color:#001b0f;margin:0 0 4px">Fire Risk Factors</h2>
                <p style="font-size:13px;color:#414943;margin:0">Real-time environmental data for {selected}</p>
            </div>
            <span class="risk-badge {'badge-high' if fire_lvl == 'HIGH' else 'badge-moderate' if fire_lvl == 'MODERATE' else 'badge-low'}">{fire_lvl}</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">🏢 Building density</span>
            <span class="factor-value {'high' if info['category'] in ['Informal Settlement','Dense Mixed'] else 'normal'}">{info['category']}</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">⏱️ Last incident reported</span>
            <span class="factor-value normal">3 days ago</span>
        </div>
        <div class="factor-row">
            <span class="factor-label">🚒 Fire service response</span>
            <span class="factor-value {'high' if info['category'] == 'Informal Settlement' else 'normal'}">{'~45 minutes' if info['category'] == 'Informal Settlement' else '~25 minutes' if info['category'] == 'Dense Mixed' else '~15 minutes'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Seasonal Fire Risk Trend")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base = RISK_SCORE[info["fire_risk"]]
    seasonal = [base+15, base+20, base+10, base-5, base-15, base-20, base-25, base-22, base-15, base-5, base+8, base+18]
    seasonal = [max(10, min(95, s)) for s in seasonal]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=months, y=seasonal, fill="tozeroy",
        fillcolor="rgba(255,100,0,0.2)", line=dict(color="#ff6400", width=2.5), mode="lines+markers"))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig2.update_layout(xaxis_title="Month", yaxis_title="Risk Score (%)", plot_bgcolor="white", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🗺️ Fire Risk Heatmap")
    fm2 = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
    for d, dinfo in ACCRA_DISTRICTS.items():
        folium.CircleMarker(
            location=[dinfo["lat"], dinfo["lon"]], radius=14,
            color=RISK_FOLIUM[dinfo["fire_risk"]], fill=True, fill_opacity=0.55,
            popup=f"{d}: {dinfo['fire_risk']}", tooltip=d
        ).add_to(fm2)
    folium.Marker(location=[lat, lon], popup=f"{selected} — {fire_lvl}",
        icon=folium.Icon(color="red", icon="fire")).add_to(fm2)
    st_folium(fm2, width=None, height=380, returned_objects=[])

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin:24px 0 12px">
        <p class="section-header" style="margin:0">⚠️ Community Hazard Feed</p>
        <span style="background:#ffbf00;color:#261a00;font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    hazards = [
        ("Exposed wiring near Nima Market", "Reported at Gate 4 near food stalls. High risk of sparking during rain.", "2 hours ago", 12),
        ("Overheating transformer on Kaneshie main road", "Visible sparking reported. Keep clear of the area.", "5 hours ago", 8),
        ("Illegal fuel storage at Chorkor junction", "Multiple containers near market entrance. Fire hazard.", "1 day ago", 24),
    ]
    for title, desc, time, confirms in hazards:
        st.markdown(f"""
        <div class="hazard-item">
            <div class="hazard-header">
                <div class="hazard-icon">⚡</div>
                <div>
                    <h3 style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:15px;color:#001b0f;margin:0 0 4px">{title}</h3>
                    <p style="font-size:13px;color:#414943;margin:0">{desc}</p>
                </div>
            </div>
            <div class="hazard-footer">
                <span style="font-size:11px;color:#94a3b8;font-weight:700">{time}</span>
                <button class="confirm-btn">CONFIRM · {confirms}</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<button class="danger-action-btn">⚡ Report an electrical hazard</button>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WASTE & DRAINAGE — friend's calendar + all original features
# ─────────────────────────────────────────────
elif page == "🗑️ Waste & Drainage":
    st.markdown(f"## 🗑️ Waste & Drainage — {selected}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Gutters blocked nearby", "4 reports" if waste_lvl == "HIGH" else "1 report" if waste_lvl == "MODERATE" else "0 reports")
    col2.metric("Resolved this week", "12" if waste_lvl == "HIGH" else "8" if waste_lvl == "MODERATE" else "3")
    col3.metric("Cleanliness score", "42/100" if waste_lvl == "HIGH" else "67/100" if waste_lvl == "MODERATE" else "88/100")

    st.markdown('<p class="section-header">📅 Collection Schedule</p>', unsafe_allow_html=True)
    days_short = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    today_idx = datetime.now().weekday()
    tomorrow_idx = (today_idx + 1) % 7
    today_date = datetime.now()

    cal_html = '<div class="calendar-strip">'
    for i, day in enumerate(days_short):
        date_num = (today_date + timedelta(days=i-today_idx)).day
        cls = "today" if i == today_idx else "collection" if i == tomorrow_idx else ""
        icon = "🚛" if i == tomorrow_idx else ""
        cal_html += f'<div class="cal-day {cls}"><span class="day-name">{day}</span><span class="day-num">{date_num}</span>{"<span style=font-size:14px>"+icon+"</span>" if icon else ""}</div>'
    cal_html += '</div>'
    st.markdown(cal_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#bdedd2;border-radius:16px;padding:16px;margin:12px 0;display:flex;align-items:center;gap:12px">
        <div style="width:40px;height:40px;border-radius:999px;background:#002113;display:flex;align-items:center;justify-content:center;color:#bdedd2;font-size:18px">⏰</div>
        <div>
            <p style="font-size:13px;color:#002113;margin:0">Next Collection</p>
            <p style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:15px;color:#002113;margin:0">Tomorrow • 07:00 AM</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    blocked = 4 if waste_lvl == "HIGH" else 1 if waste_lvl == "MODERATE" else 0
    resolved = 12 if waste_lvl == "HIGH" else 8 if waste_lvl == "MODERATE" else 3
    score = "42/100" if waste_lvl == "HIGH" else "67/100" if waste_lvl == "MODERATE" else "88/100"
    score_color = "red" if waste_lvl == "HIGH" else "amber" if waste_lvl == "MODERATE" else "green"

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-tile"><div class="stat-val red">{blocked}</div><div class="stat-label">Blocked Nearby</div></div>
        <div class="stat-tile"><div class="stat-val green">{resolved}</div><div class="stat-label">Resolved Week</div></div>
        <div class="stat-tile"><div class="stat-val {score_color}">{score}</div><div class="stat-label">Area Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Blockage Reports — Last 7 Days")
    dates_w = [(datetime.now() - timedelta(days=i)).strftime("%d %b") for i in range(6, -1, -1)]
    blockages = [8,6,9,12,7,10,4] if waste_lvl == "HIGH" else [3,2,4,5,3,4,2] if waste_lvl == "MODERATE" else [0,1,0,1,0,1,0]
    fig3 = px.bar(x=dates_w, y=blockages, color=blockages,
        color_continuous_scale=["green", "yellow", "red"],
        labels={"x": "Date", "y": "Blockage Reports"})
    fig3.update_layout(plot_bgcolor="white", height=320, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🗺️ Drainage Map")
    wm = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    blockages_pts = [
        {"lat": lat+0.005, "lon": lon+0.003, "name": "Main road junction", "sev": "High", "n": 7},
        {"lat": lat-0.004, "lon": lon+0.006, "name": "Market entrance", "sev": "Moderate", "n": 12},
        {"lat": lat+0.002, "lon": lon-0.005, "name": "School street", "sev": "Low", "n": 3},
    ]
    for p in blockages_pts:
        c = "red" if p["sev"] == "High" else "orange" if p["sev"] == "Moderate" else "green"
        folium.CircleMarker(location=[p["lat"], p["lon"]], radius=10,
            color=c, fill=True, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{p['name']}</b><br>Severity: {p['sev']}<br>Reports: {p['n']}", max_width=200),
            tooltip=p["name"]).add_to(wm)
    folium.Marker(location=[lat, lon], popup=selected,
        icon=folium.Icon(color="darkgreen", icon="home")).add_to(wm)
    st_folium(wm, width=None, height=380, returned_objects=[])

    st.markdown('<p class="section-header">📋 Recent Blockage Reports</p>', unsafe_allow_html=True)
    for p in blockages_pts:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-icon">💧</div>
            <div><h3>{p['name']}</h3><p>{p['sev']} severity · {p['n']} community confirmations</p></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px">
        <button class="primary-action-btn" style="font-size:13px;padding:14px;margin:0">🚨 Report Blockage</button>
        <button class="primary-action-btn" style="font-size:13px;padding:14px;margin:0">📅 Full Schedule</button>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REPORT HAZARD — friend's step UI + original form logic
# ─────────────────────────────────────────────
elif page == "📢 Report Hazard":
    st.markdown("## 📢 Report a Hazard")
    st.caption("Your report directly improves SmartAccra's predictions for everyone in your community.")

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin:20px 0 10px">
        <h3 style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:700;color:#001b0f;margin:0">Step 1: Select Type</h3>
        <span style="background:rgba(1,50,32,0.1);color:#013220;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:700">1 of 4</span>
    </div>
    <div class="hazard-type-grid">
        <div class="hazard-type-btn selected"><span style="font-size:32px">💧</span><span style="font-size:12px;font-weight:600;color:#001b0f;text-align:center">Blocked Gutter</span></div>
        <div class="hazard-type-btn"><span style="font-size:32px">⚡</span><span style="font-size:12px;font-weight:600;color:#001b0f;text-align:center">Electrical Hazard</span></div>
        <div class="hazard-type-btn"><span style="font-size:32px">🗑️</span><span style="font-size:12px;font-weight:600;color:#001b0f;text-align:center">Illegal Dumping</span></div>
    </div>
    """, unsafe_allow_html=True)

    htype = st.radio("Hazard type", [
        "🌊 Blocked Gutter / Drainage",
        "🔥 Electrical Hazard / Illegal Wiring",
        "🗑️ Illegal Waste Dumping",
        "🌊 Active Flooding",
        "🔥 Fire or Smoke Sighted"
    ], label_visibility="collapsed")

    st.markdown('<h3 style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:18px;font-weight:700;color:#001b0f;margin:20px 0 10px">Step 2: Location</h3>', unsafe_allow_html=True)
    rdistrict = st.selectbox("District", sorted(ACCRA_DISTRICTS.keys()),
        index=sorted(ACCRA_DISTRICTS.keys()).index(selected))
    street = st.text_input("Street name or landmark", placeholder="e.g. Near Nima Market, Station Road...")

    st.markdown('<h3 style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:18px;font-weight:700;color:#001b0f;margin:20px 0 10px">Step 3: Severity Level</h3>', unsafe_allow_html=True)
    sev = st.slider("Severity (1 = minor, 5 = severe)", 1, 5, 3)
    sev_label = {1: "Minor", 2: "Mild", 3: "Moderate", 4: "Serious", 5: "Severe"}
    st.markdown(f'<p style="text-align:center;font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:600;color:#795900;font-size:15px">Level {sev}: {sev_label[sev]}</p>', unsafe_allow_html=True)

    st.markdown('<h3 style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:18px;font-weight:700;color:#001b0f;margin:20px 0 10px">Step 4: Additional Details</h3>', unsafe_allow_html=True)
    desc = st.text_area("Additional details (optional)", placeholder="What did you see?")

    if st.button("🚨 Submit Report", use_container_width=True):
        st.success(f"""
        ✅ **Report submitted successfully!**

        {htype} reported in **{rdistrict}**
        Severity: {sev_label[sev]}

        Thank you, {user_name}! Your report is live and helping protect your community. 🙏
        """)
        st.balloons()

    st.markdown("---")
    st.markdown('<p class="section-header">📊 Community Impact This Month</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total reports", "847")
    col2.metric("Resolved", "623")
    col3.metric("Floods prevented", "12")

# ─────────────────────────────────────────────
# ALERTS — friend's coloured feed design
# ─────────────────────────────────────────────
elif page == "🔔 Alerts":
    st.markdown(f"## 🔔 Your Alerts")
    st.markdown(f"*Real-time community and safety notifications for {selected}*")

    alerts = [
        ("flood", "🌊", "FLOOD RISK", "#3b82f6",
         f"Moderate flood risk expected in {selected}",
         "Heavy downpour expected. Residents advised to clear drainage and move valuables to higher ground.", "12m ago"),
        ("fire", "🔥", "ELECTRICAL HAZARD", "#ef4444",
         "Electrical hazard confirmed",
         "Faulty transformer reported near Makola Market. ECG teams on-site. Please maintain safe distance.", "1h ago"),
        ("waste", "🗑️", "WASTE COLLECTION", "#f59e0b",
         "Borla collection rescheduled",
         "Weekly collection moved to Thursday morning due to fleet maintenance. Ensure bins are accessible.", "4h ago"),
        ("resolved", "✅", "RESOLVED", "#10b981",
         "Road clearance complete",
         "The blockage on Independence Avenue has been cleared. Traffic flow has returned to normal.", "Yesterday"),
    ]

    for cls, icon, tag, color, title, desc, time in alerts:
        st.markdown(f"""
        <div class="alert-item {cls}">
            <div class="alert-icon {cls}">{icon}</div>
            <div style="flex:1">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:11px;font-weight:700;color:{color};letter-spacing:0.05em">{tag}</span>
                    <span style="font-size:12px;color:#94a3b8">{time}</span>
                </div>
                <h3 style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:15px;color:#001b0f;margin:0 0 4px">{title}</h3>
                <p style="font-size:13px;color:#414943;margin:0;line-height:1.5">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ABOUT — original content preserved
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("## About SmartAccra")
    st.markdown("""
    **SmartAccra** is a data-driven urban risk intelligence platform built to address SDG 11 — Sustainable Cities and Communities, with specific focus on Accra, Ghana.

    ### The Problem
    Accra's informal settlements face three preventable, recurring crises:
    - 🌊 **Flooding** — Blocked gutters and poor drainage during rainy season
    - 🔥 **Fires** — Densely packed buildings with informal/illegal wiring
    - 🗑️ **Waste accumulation** — Directly worsens flooding by clogging drains

    Residents currently have **no warning system** and **no data tools** to anticipate these disasters.

    ### Our Approach

    | Module | Data Source | Method |
    |---|---|---|
    | 🌊 Flood Risk | Open-Meteo Weather API (real-time) | Dynamic risk scoring + rainfall |
    | 🔥 Fire Risk | Building density + infrastructure data | Logistic Regression classification |
    | 🗑️ Waste & Drainage | Community-reported data | Descriptive analytics + trends |

    ### Risk Profiling
    - 🔴 **Informal Settlements** — Highest risk (Nima, Chorkor, Old Fadama, Agbogbloshie...)
    - 🟠 **Dense Mixed Areas** — Moderate-high (Kaneshie, Lapaz, Darkuman...)
    - 🟡 **Working Class Suburbs** — Moderate (Adenta, Madina, Dansoman...)
    - 🟢 **Affluent Planned Areas** — Low risk (East Legon, Cantonments, Airport Residential...)

    ### Roadmap
    - **Phase 1 (current MVP)**: Risk dashboards, predictions, community reporting
    - **Phase 2**: Google OAuth, SMS alerts, push notifications
    - **Phase 3**: Power outage prediction (dumsor), expanded coverage to other Ghanaian cities

    Built by **Sally Annan** as part of a data products course project.
    """)
