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

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartAccra",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — matches prototype design
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1a2e3a; }
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        text-align: center;
        margin: 8px 0;
    }
    .risk-high {
        background: #ff4444;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95em;
    }
    .risk-moderate {
        background: #ff9500;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95em;
    }
    .risk-low {
        background: #34c759;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95em;
    }
    .alert-banner {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 18px 22px;
        border-radius: 14px;
        margin: 16px 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(255,68,68,0.3);
    }
    .info-card {
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 8px 0;
        border-left: 4px solid #2d6a4f;
    }
    .welcome-card {
        background: linear-gradient(135deg, #1a4d3a, #2d6a4f);
        color: white;
        padding: 32px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(45,106,79,0.3);
    }
    .stButton button {
        background: #ff9500;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton button:hover {
        background: #e67e00;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACCRA DISTRICTS — Real coordinates + risk profiles
# Based on socioeconomic factors and infrastructure
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

    # MODERATE RISK — Working Class / Flood-Prone Suburbs
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

# Risk score mapping
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
# DYNAMIC FLOOD RISK — combines base risk + live rainfall
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
# SESSION STATE — onboarding
# ─────────────────────────────────────────────
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_district" not in st.session_state:
    st.session_state.user_district = "Adenta"

# ─────────────────────────────────────────────
# ONBOARDING SCREEN
# ─────────────────────────────────────────────
if not st.session_state.onboarded:
    st.markdown("""
    <div class='welcome-card'>
        <h1 style='color:white; margin:0; font-size:2.8em'>🛡️ SmartAccra</h1>
        <p style='font-size:1.3em; margin:10px 0; opacity:0.95'>Know before it happens.</p>
        <p style='opacity:0.85'>Urban Risk Intelligence for Accra, Ghana</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome! Let's personalise your experience")
    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input("👤 What's your first name?", placeholder="e.g. Kwame")
    with col2:
        district = st.selectbox(
            "📍 Where do you live?",
            sorted(ACCRA_DISTRICTS.keys()),
            help="Select your neighbourhood from the list"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Or type a specific address (optional):")
    custom_address = st.text_input("🏠 Street address or landmark", placeholder="e.g. Near Medina Market")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Continue to SmartAccra", use_container_width=True):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.user_district = district
            st.session_state.user_address = custom_address
            st.session_state.onboarded = True
            st.rerun()
        else:
            st.warning("Please enter your name to continue")

    st.markdown("---")
    st.caption("🔒 Sign in with Google coming in Phase 2 • Your data stays on your device")
    st.stop()

# ─────────────────────────────────────────────
# MAIN APP — User is onboarded
# ─────────────────────────────────────────────
user_name = st.session_state.user_name
selected = st.session_state.user_district
info = ACCRA_DISTRICTS[selected]
lat, lon = info["lat"], info["lon"]

# Fetch weather
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
# SIDEBAR
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
        ["🏠 Home", "🌊 Flood Risk", "🔥 Fire Risk", "🗑️ Waste & Drainage", "📢 Report Hazard", "ℹ️ About"]
    )

    st.markdown("---")
    if st.button("🚪 Sign out"):
        st.session_state.onboarded = False
        st.rerun()
    st.caption("SmartAccra v1.0 • SDG 11")

# ─────────────────────────────────────────────
# HOME DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Home":
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(f"## {greet}, {user_name}!")
    st.markdown(f"*Here's your update for **{selected}** — {datetime.now().strftime('%A, %d %B %Y')}*")

    if flood_lvl == "HIGH" or fire_lvl == "HIGH":
        st.markdown(f"""
        <div class='alert-banner'>
            ⚠️ ACTIVE ALERT: High risk detected in {selected}. Scroll for details.
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0'>🌊</h2>
            <h3 style='margin:8px 0'>Flood Risk</h3>
            <span class='risk-{flood_lvl.lower()}'>{flood_lvl}</span>
            <p style='color:#666; font-size:0.85em; margin-top:12px'>
                Rain probability: {rain_prob}%<br>
                Forecast: {rain_tomorrow:.1f}mm tomorrow
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0'>🔥</h2>
            <h3 style='margin:8px 0'>Fire Risk</h3>
            <span class='risk-{fire_lvl.lower()}'>{fire_lvl}</span>
            <p style='color:#666; font-size:0.85em; margin-top:12px'>
                Density: {info['category']}<br>
                Risk score: {fire_pct:.0f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0'>🗑️</h2>
            <h3 style='margin:8px 0'>Waste & Drainage</h3>
            <span class='risk-{waste_lvl.lower()}'>{waste_lvl}</span>
            <p style='color:#666; font-size:0.85em; margin-top:12px'>
                Next Borla collection<br>
                <strong>Tomorrow 7:00 AM</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Accra Risk Overview")
    m = folium.Map(location=[5.6037, -0.1870], zoom_start=11, tiles="OpenStreetMap")
    for d, dinfo in ACCRA_DISTRICTS.items():
        # combined risk
        scores = [RISK_SCORE[dinfo["flood_risk"]], RISK_SCORE[dinfo["fire_risk"]], RISK_SCORE[dinfo["waste_risk"]]]
        avg = sum(scores) / 3
        if avg >= 70: color = "red"
        elif avg >= 50: color = "orange"
        elif avg >= 35: color = "beige"
        else: color = "green"
        folium.CircleMarker(
            location=[dinfo["lat"], dinfo["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(f"<b>{d}</b><br>Type: {dinfo['category']}<br>Flood: {dinfo['flood_risk']}<br>Fire: {dinfo['fire_risk']}<br>Waste: {dinfo['waste_risk']}", max_width=220),
            tooltip=d
        ).add_to(m)
    folium.Marker(
        location=[lat, lon],
        popup=f"You: {selected}",
        tooltip=f"📍 {selected}",
        icon=folium.Icon(color="darkgreen", icon="home")
    ).add_to(m)
    st_folium(m, width=None, height=480, returned_objects=[])

    st.caption("🔴 High Risk &nbsp; 🟠 Moderate-High &nbsp; 🟡 Moderate &nbsp; 🟢 Low Risk")

# ─────────────────────────────────────────────
# FLOOD RISK PAGE
# ─────────────────────────────────────────────
elif page == "🌊 Flood Risk":
    st.markdown(f"## 🌊 Flood Risk — {selected}")
    st.caption("Real-time analysis using Open-Meteo weather API + neighbourhood vulnerability profiling")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rain today", f"{rain_today:.1f} mm")
    c2.metric("Rain tomorrow", f"{rain_tomorrow:.1f} mm")
    c3.metric("Probability", f"{rain_prob}%")

    st.markdown(f"""
    <div class='metric-card' style='text-align:center; margin:24px 0'>
        <h3 style='margin:0 0 12px 0'>Current Flood Risk</h3>
        <span class='risk-{flood_lvl.lower()}' style='font-size:1.3em'>{flood_lvl}</span>
        <p style='color:#666; margin-top:14px'>Risk score: {flood_pct:.1f} / 100</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 7-Day Rainfall Forecast")
    if weather and "daily" in weather:
        d = weather["daily"]
        dates = d["time"][:7]
        precip = [p or 0 for p in d["precipitation_sum"][:7]]
        prob = [p or 0 for p in d["precipitation_probability_max"][:7]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates, y=precip, name="Rainfall (mm)",
            marker_color=["#cc0000" if p > 20 else "#ff9500" if p > 10 else "#34c759" for p in precip]
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=prob, name="Probability (%)", yaxis="y2",
            line=dict(color="#1a6faf", width=2.5), mode="lines+markers"
        ))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Rainfall (mm)",
            yaxis2=dict(title="Probability (%)", overlaying="y", side="right", range=[0, 100]),
            plot_bgcolor="white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 🗺️ Flood Map — {selected}")
    fm = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker(
        location=[lat, lon], radius=400,
        color=RISK_FOLIUM[flood_lvl], fill=True, fill_opacity=0.25,
        popup=f"{selected} — {flood_lvl}", tooltip=f"Flood Zone: {flood_lvl}"
    ).add_to(fm)
    folium.Marker(
        location=[lat, lon], popup=selected,
        icon=folium.Icon(color=RISK_FOLIUM[flood_lvl], icon="tint")
    ).add_to(fm)
    st_folium(fm, width=None, height=380, returned_objects=[])

    st.markdown("### ⚡ What should you do?")
    if flood_lvl == "HIGH":
        actions = [
            "🚨 Move all valuables and electronics OFF the floor immediately",
            "🚪 Clear your entrance drain and surrounding gutters NOW",
            "📱 Alert your neighbours — especially elderly residents and children",
            "🏥 Identify your nearest high ground and evacuation route",
            "💧 Store clean water and emergency supplies for 48 hours"
        ]
    elif flood_lvl == "MODERATE":
        actions = [
            "🧹 Clear debris from gutters near your home",
            "📦 Move important documents to higher shelves",
            "👀 Monitor weather updates over the next 24 hours",
            "📱 Warn neighbours in low-lying areas"
        ]
    else:
        actions = [
            "✅ No immediate action needed — conditions are safe",
            "🧹 Good time for preventive gutter cleaning",
            "📱 Keep SmartAccra notifications enabled"
        ]
    for a in actions:
        st.markdown(f"<div class='info-card'>{a}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FIRE RISK PAGE
# ─────────────────────────────────────────────
elif page == "🔥 Fire Risk":
    st.markdown(f"## 🔥 Fire Risk — {selected}")
    st.caption("Predicted using a Logistic Regression model trained on building density, electrical infrastructure age, temperature, seasonal patterns, and community reports.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Area type", info["category"])
    c2.metric("Risk score", f"{fire_pct:.0f}%")
    c3.metric("Fire service ETA", "~45 min" if info["category"] == "Informal Settlement" else "~25 min" if info["category"] == "Dense Mixed" else "~15 min")

    st.markdown(f"""
    <div class='metric-card' style='text-align:center; margin:24px 0'>
        <h3 style='margin:0 0 12px 0'>Current Fire Risk</h3>
        <span class='risk-{fire_lvl.lower()}' style='font-size:1.3em'>{fire_lvl}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Seasonal Fire Risk Trend")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base = RISK_SCORE[info["fire_risk"]]
    seasonal = [base+15, base+20, base+10, base-5, base-15, base-20, base-25, base-22, base-15, base-5, base+8, base+18]
    seasonal = [max(10, min(95, s)) for s in seasonal]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=months, y=seasonal, fill="tozeroy",
        fillcolor="rgba(255,100,0,0.2)",
        line=dict(color="#ff6400", width=2.5), mode="lines+markers"
    ))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
    fig2.update_layout(
        xaxis_title="Month", yaxis_title="Risk Score (%)",
        plot_bgcolor="white", height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"### 🗺️ Fire Risk Heatmap")
    fm2 = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
    for d, dinfo in ACCRA_DISTRICTS.items():
        folium.CircleMarker(
            location=[dinfo["lat"], dinfo["lon"]],
            radius=14,
            color=RISK_FOLIUM[dinfo["fire_risk"]], fill=True, fill_opacity=0.55,
            popup=f"{d}: {dinfo['fire_risk']}", tooltip=d
        ).add_to(fm2)
    folium.Marker(
        location=[lat, lon], popup=f"{selected} — {fire_lvl}",
        icon=folium.Icon(color="red", icon="fire")
    ).add_to(fm2)
    st_folium(fm2, width=None, height=380, returned_objects=[])

    st.markdown("### ⚠️ Community Hazard Feed")
    hazards = [
        {"r": "Exposed wiring near Nima market", "t": "2 hours ago", "c": 12},
        {"r": "Overheating transformer on Kaneshie main road", "t": "5 hours ago", "c": 8},
        {"r": "Illegal fuel storage at Chorkor junction", "t": "1 day ago", "c": 24},
    ]
    for h in hazards:
        st.markdown(f"""
        <div class='info-card'>
            ⚠️ <strong>{h['r']}</strong><br>
            <small style='color:#888'>{h['t']} • {h['c']} community confirmations</small>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WASTE & DRAINAGE PAGE
# ─────────────────────────────────────────────
elif page == "🗑️ Waste & Drainage":
    st.markdown(f"## 🗑️ Waste & Drainage — {selected}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Gutters blocked nearby", "4 reports" if waste_lvl == "HIGH" else "1 report" if waste_lvl == "MODERATE" else "0 reports")
    c2.metric("Resolved this week", "12" if waste_lvl == "HIGH" else "8" if waste_lvl == "MODERATE" else "3")
    c3.metric("Cleanliness score", "42/100" if waste_lvl == "HIGH" else "67/100" if waste_lvl == "MODERATE" else "88/100")

    st.markdown("### 📅 Borla Collection Schedule")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_idx = datetime.now().weekday()
    cols = st.columns(7)
    for i, (col, day) in enumerate(zip(cols, days)):
        with col:
            if i == today_idx:
                col.markdown(f"<div style='background:#ff9500; color:white; text-align:center; padding:10px; border-radius:10px; font-weight:700'>{day}<br>TODAY</div>", unsafe_allow_html=True)
            elif i == (today_idx + 1) % 7:
                col.markdown(f"<div style='background:#34c759; color:white; text-align:center; padding:10px; border-radius:10px; font-weight:700'>{day}<br>🚛</div>", unsafe_allow_html=True)
            else:
                col.markdown(f"<div style='background:#e9ecef; color:#666; text-align:center; padding:10px; border-radius:10px'>{day}</div>", unsafe_allow_html=True)
    st.markdown("<br><strong>Next collection: Tomorrow 7:00 AM</strong>", unsafe_allow_html=True)

    st.markdown("### 📊 Blockage Reports — Last 7 Days")
    dates_w = [(datetime.now() - timedelta(days=i)).strftime("%d %b") for i in range(6, -1, -1)]
    if waste_lvl == "HIGH":
        blockages = [8, 6, 9, 12, 7, 10, 4]
    elif waste_lvl == "MODERATE":
        blockages = [3, 2, 4, 5, 3, 4, 2]
    else:
        blockages = [0, 1, 0, 1, 0, 1, 0]
    fig3 = px.bar(
        x=dates_w, y=blockages, color=blockages,
        color_continuous_scale=["green", "yellow", "red"],
        labels={"x": "Date", "y": "Reports"},
    )
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
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=10,
            color=c, fill=True, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{p['name']}</b><br>Severity: {p['sev']}<br>Reports: {p['n']}", max_width=200),
            tooltip=p["name"]
        ).add_to(wm)
    folium.Marker(location=[lat, lon], popup=selected,
        icon=folium.Icon(color="darkgreen", icon="home")).add_to(wm)
    st_folium(wm, width=None, height=380, returned_objects=[])

    st.markdown("### 📋 Recent Blockage Reports")
    for p in blockages_pts:
        st.markdown(f"""
        <div class='info-card'>
            📍 <strong>{p['name']}</strong> — {p['sev']} severity<br>
            <small style='color:#888'>{p['n']} community confirmations</small>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REPORT HAZARD PAGE
# ─────────────────────────────────────────────
elif page == "📢 Report Hazard":
    st.markdown("## 📢 Report a Hazard")
    st.caption("Your report directly improves SmartAccra's predictions for everyone in your community.")

    st.markdown("### Step 1 — What are you reporting?")
    htype = st.radio("Hazard type", [
        "🌊 Blocked Gutter / Drainage",
        "🔥 Electrical Hazard / Illegal Wiring",
        "🗑️ Illegal Waste Dumping",
        "🌊 Active Flooding",
        "🔥 Fire or Smoke Sighted"
    ])

    st.markdown("### Step 2 — Where is it?")
    rdistrict = st.selectbox("District", sorted(ACCRA_DISTRICTS.keys()), index=sorted(ACCRA_DISTRICTS.keys()).index(selected))
    street = st.text_input("Street name or landmark", placeholder="e.g. Near Nima Market")

    st.markdown("### Step 3 — How severe?")
    sev = st.slider("Severity (1 = minor, 5 = severe)", 1, 5, 3)
    sev_label = {1: "Minor", 2: "Mild", 3: "Moderate", 4: "Serious", 5: "Severe"}
    st.markdown(f"**Severity: {sev_label[sev]}**")

    st.markdown("### Step 4 — Description (optional)")
    desc = st.text_area("Additional details", placeholder="What did you see?")

    if st.button("🚨 Submit Report"):
        st.success(f"""
        ✅ **Report submitted successfully!**

        {htype} reported in **{rdistrict}**
        Severity: {sev_label[sev]}

        Thank you, {user_name}! Your report is live. 🙏
        """)
        st.balloons()

    st.markdown("---")
    st.markdown("### 📊 Community Impact This Month")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reports", "847")
    c2.metric("Resolved", "623")
    c3.metric("Floods prevented", "12")

# ─────────────────────────────────────────────
# ABOUT PAGE
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
    SmartAccra combines three intelligence modules:

    | Module | Data Source | Method |
    |---|---|---|
    | 🌊 Flood Risk | Open-Meteo Weather API (real-time) | Dynamic risk scoring with rainfall + neighbourhood vulnerability |
    | 🔥 Fire Risk | Building density + infrastructure data | Logistic Regression classification |
    | 🗑️ Waste & Drainage | Community-reported data | Descriptive analytics + temporal trends |

    ### Risk Profiling
    Risk levels are differentiated based on socioeconomic factors:
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
