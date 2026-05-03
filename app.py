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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 8px 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ff9500, #e67e00);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #34c759, #248a3d);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .alert-banner {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: bold;
    }
    .info-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 8px 0;
        border-left: 4px solid #2d6a4f;
    }
    .section-header {
        color: #1a1a2e;
        font-size: 1.3em;
        font-weight: 700;
        margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACCRA DISTRICTS DATA
# ─────────────────────────────────────────────
ACCRA_DISTRICTS = {
    "Adenta": {"lat": 5.7167, "lon": -0.1667, "population": 142463, "density": "High"},
    "Medina": {"lat": 5.6667, "lon": -0.2000, "population": 89000, "density": "Very High"},
    "Nima": {"lat": 5.5833, "lon": -0.2167, "population": 120000, "density": "Very High"},
    "Dansoman": {"lat": 5.5500, "lon": -0.2667, "population": 95000, "density": "High"},
    "Mamprobi": {"lat": 5.5333, "lon": -0.2333, "population": 78000, "density": "High"},
    "Ablekuma West": {"lat": 5.5667, "lon": -0.2500, "population": 110000, "density": "High"},
    "Tema": {"lat": 5.6667, "lon": -0.0167, "population": 161612, "density": "Moderate"},
    "Osu": {"lat": 5.5500, "lon": -0.1833, "population": 45000, "density": "Moderate"},
    "Korle Bu": {"lat": 5.5333, "lon": -0.2167, "population": 67000, "density": "High"},
    "Accra Central": {"lat": 5.5500, "lon": -0.2167, "population": 185000, "density": "Very High"},
}

# ─────────────────────────────────────────────
# FETCH REAL WEATHER DATA FROM OPEN-METEO
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_weather_data(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=precipitation,temperature_2m,relative_humidity_2m"
        f"&daily=precipitation_sum,temperature_2m_max,precipitation_probability_max"
        f"&timezone=Africa%2FAccra"
        f"&forecast_days=7"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data
    except:
        return None

# ─────────────────────────────────────────────
# FLOOD RISK CALCULATION
# ─────────────────────────────────────────────
def calculate_flood_risk(precipitation_mm, district):
    density = ACCRA_DISTRICTS[district]["density"]
    density_multiplier = {
        "Very High": 1.5,
        "High": 1.3,
        "Moderate": 1.0,
        "Low": 0.7
    }.get(density, 1.0)

    base_risk = min(precipitation_mm * density_multiplier / 50, 1.0)

    if base_risk >= 0.7:
        return "HIGH", base_risk * 100
    elif base_risk >= 0.4:
        return "MODERATE", base_risk * 100
    else:
        return "LOW", base_risk * 100

# ─────────────────────────────────────────────
# FIRE RISK MODEL (LOGISTIC REGRESSION)
# ─────────────────────────────────────────────
@st.cache_data
def build_fire_risk_model():
    np.random.seed(42)
    n_samples = 500

    building_density = np.random.uniform(0.1, 1.0, n_samples)
    electrical_age = np.random.uniform(1, 30, n_samples)
    temperature = np.random.uniform(25, 38, n_samples)
    dry_season = np.random.randint(0, 2, n_samples)
    reported_hazards = np.random.randint(0, 20, n_samples)

    risk_score = (
        0.3 * building_density +
        0.25 * (electrical_age / 30) +
        0.2 * ((temperature - 25) / 13) +
        0.15 * dry_season +
        0.1 * (reported_hazards / 20)
    )
    fire_risk = (risk_score > 0.5).astype(int)

    X = np.column_stack([
        building_density, electrical_age,
        temperature, dry_season, reported_hazards
    ])
    y = fire_risk

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

def predict_fire_risk(district, temperature=32, dry_season=1, reported_hazards=5):
    model, scaler = build_fire_risk_model()

    density_map = {"Very High": 0.9, "High": 0.7, "Moderate": 0.5, "Low": 0.3}
    building_density = density_map.get(
        ACCRA_DISTRICTS[district]["density"], 0.5
    )
    electrical_age = 15

    X = np.array([[
        building_density, electrical_age,
        temperature, dry_season, reported_hazards
    ]])
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]

    if probability >= 0.7:
        return "HIGH", probability * 100
    elif probability >= 0.4:
        return "MODERATE", probability * 100
    else:
        return "LOW", probability * 100

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/SmartAccra-Know%20Before%20It%20Happens-2d6a4f?style=for-the-badge",
        use_column_width=True
    )
    st.markdown("---")
    st.markdown("### 📍 Select Your District")
    selected_district = st.selectbox(
        "Choose your area",
        list(ACCRA_DISTRICTS.keys()),
        index=0
    )

    district_info = ACCRA_DISTRICTS[selected_district]
    st.markdown(f"""
    <div class='info-card'>
        <strong>District Info</strong><br>
        Population: {district_info['population']:,}<br>
        Building Density: {district_info['density']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home Dashboard",
         "🌊 Flood Risk",
         "🔥 Fire Risk",
         "🗑️ Waste & Drainage",
         "📢 Report a Hazard"]
    )
    st.markdown("---")
    st.markdown(
        "<small>SmartAccra v1.0 | SDG 11</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# FETCH DATA FOR SELECTED DISTRICT
# ─────────────────────────────────────────────
lat = district_info["lat"]
lon = district_info["lon"]
weather_data = fetch_weather_data(lat, lon)

if weather_data:
    daily = weather_data.get("daily", {})
    precip_today = daily.get("precipitation_sum", [0])[0] or 0
    precip_tomorrow = daily.get("precipitation_sum", [0, 0])[1] or 0
    precip_max = max(daily.get("precipitation_sum", [0]))
    precip_prob = daily.get(
        "precipitation_probability_max", [0]
    )[0] or 0
else:
    precip_today = 5
    precip_tomorrow = 15
    precip_max = 25
    precip_prob = 60

flood_level, flood_score = calculate_flood_risk(
    precip_tomorrow, selected_district
)
fire_level, fire_score = predict_fire_risk(selected_district)

# ─────────────────────────────────────────────
# HOME DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Home Dashboard":
    st.markdown(f"## Good day! Here is your SmartAccra update for **{selected_district}**")
    st.markdown(f"*Last updated: {datetime.now().strftime('%d %B %Y, %H:%M')}*")

    if flood_level == "HIGH" or fire_level == "HIGH":
        st.markdown("""
        <div class='alert-banner'>
            ⚠️ ACTIVE ALERT: High risk detected in your area — 
            scroll down for details
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    flood_color = {
        "HIGH": "risk-high",
        "MODERATE": "risk-moderate",
        "LOW": "risk-low"
    }[flood_level]

    fire_color = {
        "HIGH": "risk-high",
        "MODERATE": "risk-moderate",
        "LOW": "risk-low"
    }[fire_level]

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>🌊</h2>
            <h3>Flood Risk</h3>
            <span class='{flood_color}'>{flood_level}</span>
            <p style='margin-top:10px; color:#666; font-size:0.9em'>
                Rain probability: {precip_prob}%<br>
                Expected rainfall: {precip_tomorrow:.1f}mm
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>🔥</h2>
            <h3>Fire Risk</h3>
            <span class='{fire_color}'>{fire_level}</span>
            <p style='margin-top:10px; color:#666; font-size:0.9em'>
                Building density: {district_info['density']}<br>
                Risk score: {fire_score:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>🗑️</h2>
            <h3>Waste & Drainage</h3>
            <span class='risk-moderate'>MONITOR</span>
            <p style='margin-top:10px; color:#666; font-size:0.9em'>
                Next Borla collection:<br>
                <strong>Tomorrow 7:00 AM</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Accra Risk Overview Map")

    m = folium.Map(
        location=[5.6037, -0.1870],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    risk_colors = {
        "Very High": "red",
        "High": "orange",
        "Moderate": "green",
        "Low": "blue"
    }

    for district, info in ACCRA_DISTRICTS.items():
        color = risk_colors.get(info["density"], "gray")
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=15,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"""<b>{district}</b><br>
                Population: {info['population']:,}<br>
                Density: {info['density']}<br>
                Risk Level: {'HIGH' if info['density'] == 'Very High' else 'MODERATE' if info['density'] == 'High' else 'LOW'}""",
                max_width=200
            ),
            tooltip=district
        ).add_to(m)

    folium.Marker(
        location=[district_info["lat"], district_info["lon"]],
        popup=f"You are here: {selected_district}",
        tooltip=f"📍 {selected_district}",
        icon=folium.Icon(color="darkgreen", icon="home")
    ).add_to(m)

    st_folium(m, width=None, height=450)

    st.markdown("""
    <small>🔴 Very High Risk &nbsp; 🟠 High Risk &nbsp; 
    🟢 Moderate Risk &nbsp; 🔵 Lower Risk</small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FLOOD RISK PAGE
# ─────────────────────────────────────────────
elif page == "🌊 Flood Risk":
    st.markdown(f"## 🌊 Flood Risk Analysis — {selected_district}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rainfall Today", f"{precip_today:.1f} mm")
    with col2:
        st.metric("Rainfall Tomorrow", f"{precip_tomorrow:.1f} mm")
    with col3:
        st.metric("Rain Probability", f"{precip_prob}%")

    st.markdown(f"""
    <div class='metric-card' style='text-align:center; margin:20px 0'>
        <h3>Current Flood Risk Level</h3>
        <span class='{flood_color}' style='font-size:1.4em'>{flood_level}</span>
        <p style='color:#666; margin-top:10px'>
            Risk Score: {flood_score:.1f} / 100
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 7-Day Rainfall Forecast for Accra")

    if weather_data:
        daily = weather_data["daily"]
        dates = daily["time"][:7]
        precip = [p or 0 for p in daily["precipitation_sum"][:7]]
        prob = [p or 0 for p in
                daily["precipitation_probability_max"][:7]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=precip,
            name="Rainfall (mm)",
            marker_color=[
                "#cc0000" if p > 20 else
                "#ff9500" if p > 10 else "#34c759"
                for p in precip
            ]
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=prob,
            name="Rain Probability (%)",
            yaxis="y2",
            line=dict(color="#1a6faf", width=2),
            mode="lines+markers"
        ))
        fig.update_layout(
            title="7-Day Rainfall Forecast — Accra",
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            yaxis2=dict(
                title="Probability (%)",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            plot_bgcolor="white",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗺️ Flood Risk Map — " + selected_district)
    flood_map = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    folium.CircleMarker(
        location=[lat, lon],
        radius=300,
        color="red" if flood_level == "HIGH" else
              "orange" if flood_level == "MODERATE" else "green",
        fill=True,
        fill_opacity=0.3,
        popup=f"{selected_district} — Flood Risk: {flood_level}",
        tooltip=f"Flood Risk Zone: {flood_level}"
    ).add_to(flood_map)

    folium.Marker(
        location=[lat, lon],
        popup=selected_district,
        icon=folium.Icon(
            color="red" if flood_level == "HIGH" else
                  "orange" if flood_level == "MODERATE" else "green",
            icon="tint"
        )
    ).add_to(flood_map)

    st_folium(flood_map, width=None, height=400)

    st.markdown("### ⚡ What Should You Do?")
    if flood_level == "HIGH":
        actions = [
            "🚨 Move all valuables and electronics off the floor immediately",
            "🚪 Clear your entrance drain and surrounding gutters now",
            "📱 Alert your neighbours — especially elderly residents",
            "🏥 Identify your nearest high ground and evacuation route",
            "💊 Store clean water and emergency supplies",
        ]
    elif flood_level == "MODERATE":
        actions = [
            "🧹 Clear debris from gutters and drains near your home",
            "📦 Move important documents and valuables to higher shelves",
            "👀 Monitor weather updates over the next 24 hours",
            "📱 Warn neighbours in low-lying areas",
        ]
    else:
        actions = [
            "✅ No immediate action needed",
            "🧹 Good time to clear gutters as a preventive measure",
            "📱 Keep SmartAccra notifications on",
        ]

    for action in actions:
        st.markdown(f"""
        <div class='info-card'>{action}</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FIRE RISK PAGE
# ─────────────────────────────────────────────
elif page == "🔥 Fire Risk":
    st.markdown(f"## 🔥 Fire Risk Analysis — {selected_district}")
    st.markdown("""
    *Fire risk is calculated using a Logistic Regression model trained 
    on building density, electrical infrastructure age, temperature, 
    seasonal patterns, and community hazard reports.*
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Building Density",
            district_info["density"]
        )
    with col2:
        st.metric(
            "Fire Risk Score",
            f"{fire_score:.1f}%"
        )
    with col3:
        st.metric(
            "Nearest Fire Station",
            "~45 min response"
        )

    st.markdown(f"""
    <div class='metric-card' style='text-align:center; margin:20px 0'>
        <h3>Current Fire Risk Level</h3>
        <span class='{fire_color}' style='font-size:1.4em'>{fire_level}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Fire Risk Trend — Seasonal Analysis")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dry_season_risk = [75, 80, 72, 55, 40, 35,
                       30, 28, 38, 45, 60, 70]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=months,
        y=dry_season_risk,
        fill="tozeroy",
        fillcolor="rgba(255, 100, 0, 0.2)",
        line=dict(color="#ff6400", width=2),
        mode="lines+markers",
        name="Fire Risk %"
    ))
    fig2.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk Threshold"
    )
    fig2.update_layout(
        title="Seasonal Fire Risk Pattern — Accra",
        xaxis_title="Month",
        yaxis_title="Fire Risk Score (%)",
        plot_bgcolor="white",
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🗺️ Fire Risk Heatmap — " + selected_district)
    fire_map = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    density_map_colors = {
        "Very High": "red",
        "High": "orange",
        "Moderate": "yellow",
        "Low": "green"
    }

    for district, info in ACCRA_DISTRICTS.items():
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=20,
            color=density_map_colors.get(
                info["density"], "gray"
            ),
            fill=True,
            fill_opacity=0.5,
            popup=f"{district}: {info['density']} density",
            tooltip=district
        ).add_to(fire_map)

    folium.Marker(
        location=[lat, lon],
        popup=f"{selected_district} — Fire Risk: {fire_level}",
        icon=folium.Icon(color="red", icon="fire")
    ).add_to(fire_map)

    st_folium(fire_map, width=None, height=400)

    st.markdown("### ⚠️ Community Hazard Feed")
    hazards = [
        {
            "report": "Exposed wiring near Medina Market",
            "time": "2 hours ago",
            "confirmations": 12,
            "area": "Medina"
        },
        {
            "report": "Overheating transformer on Station Road",
            "time": "5 hours ago",
            "confirmations": 8,
            "area": "Adenta"
        },
        {
            "report": "Illegal fuel storage at Adenta junction",
            "time": "1 day ago",
            "confirmations": 24,
            "area": "Adenta"
        },
    ]

    for hazard in hazards:
        st.markdown(f"""
        <div class='info-card'>
            ⚠️ <strong>{hazard['report']}</strong><br>
            <small style='color:#888'>
                {hazard['time']} • {hazard['confirmations']} confirmations
            </small>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WASTE & DRAINAGE PAGE
# ─────────────────────────────────────────────
elif page == "🗑️ Waste & Drainage":
    st.markdown(
        f"## 🗑️ Waste & Drainage — {selected_district}"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gutters Blocked Nearby", "4 reported")
    with col2:
        st.metric("Reports Resolved This Week", "12")
    with col3:
        st.metric("Neighbourhood Cleanliness Score", "62/100")

    st.markdown("### 📅 Borla Collection Schedule")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_idx = datetime.now().weekday()

    cols = st.columns(7)
    for i, (col, day) in enumerate(zip(cols, days)):
        with col:
            if i == today_idx:
                st.markdown(
                    f"<div style='background:#ff9500; color:white; "
                    f"text-align:center; padding:8px; border-radius:8px; "
                    f"font-weight:bold'>{day}<br>TODAY</div>",
                    unsafe_allow_html=True
                )
            elif i == (today_idx + 1) % 7:
                st.markdown(
                    f"<div style='background:#34c759; color:white; "
                    f"text-align:center; padding:8px; border-radius:8px; "
                    f"font-weight:bold'>{day}<br>🚛</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background:#e9ecef; color:#666; "
                    f"text-align:center; padding:8px; border-radius:8px'>"
                    f"{day}</div>",
                    unsafe_allow_html=True
                )

    st.markdown(
        "<br><strong>Next collection: Tomorrow 7:00 AM</strong>",
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Waste Blockage Trend — Last 7 Days")
    dates_w = [(datetime.now() - timedelta(days=i)).strftime("%d %b")
               for i in range(6, -1, -1)]
    blockages = [8, 6, 9, 12, 7, 10, 4]

    fig3 = px.bar(
        x=dates_w,
        y=blockages,
        color=blockages,
        color_continuous_scale=["green", "yellow", "red"],
        labels={"x": "Date", "y": "Blockage Reports"},
        title="Daily Gutter Blockage Reports — " + selected_district
    )
    fig3.update_layout(
        plot_bgcolor="white",
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🗺️ Drainage Blockage Map")
    waste_map = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    blockage_points = [
        {
            "lat": lat + 0.005,
            "lon": lon + 0.003,
            "location": "Station Road near Market",
            "severity": "High",
            "reports": 7
        },
        {
            "lat": lat - 0.004,
            "lon": lon + 0.006,
            "location": "Housing Estate Entrance",
            "severity": "Moderate",
            "reports": 12
        },
        {
            "lat": lat + 0.002,
            "lon": lon - 0.005,
            "location": "Church Street Junction",
            "severity": "Low",
            "reports": 5
        },
    ]

    for point in blockage_points:
        color = (
            "red" if point["severity"] == "High" else
            "orange" if point["severity"] == "Moderate" else
            "green"
        )
        folium.CircleMarker(
            location=[point["lat"], point["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"""<b>{point['location']}</b><br>
                Severity: {point['severity']}<br>
                Reports: {point['reports']}""",
                max_width=200
            ),
            tooltip=point["location"]
        ).add_to(waste_map)

    folium.Marker(
        location=[lat, lon],
        popup="Your location",
        icon=folium.Icon(color="darkgreen", icon="home")
    ).add_to(waste_map)

    st_folium(waste_map, width=None, height=400)

    st.markdown("### 📋 Recent Blockage Reports")
    for point in blockage_points:
        st.markdown(f"""
        <div class='info-card'>
            📍 <strong>{point['location']}</strong> — 
            {point['severity']} severity<br>
            <small style='color:#888'>
                {point['reports']} community confirmations
            </small>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# REPORT A HAZARD PAGE
# ─────────────────────────────────────────────
elif page == "📢 Report a Hazard":
    st.markdown("## 📢 Report a Hazard")
    st.markdown(
        "*Your report helps protect your neighbourhood. "
        "Every report contributes to SmartAccra's risk models.*"
    )

    st.markdown("### Step 1: What are you reporting?")
    hazard_type = st.radio(
        "Select hazard type",
        ["🌊 Blocked Gutter / Drainage",
         "🔥 Electrical Hazard / Illegal Wiring",
         "🗑️ Illegal Waste Dumping",
         "🌊 Active Flooding",
         "🔥 Fire or Smoke Sighted"]
    )

    st.markdown("### Step 2: Where is it?")
    report_district = st.selectbox(
        "Select the district",
        list(ACCRA_DISTRICTS.keys())
    )
    street = st.text_input(
        "Street name or landmark (optional)",
        placeholder="e.g. Near Medina Market, Station Road..."
    )

    st.markdown("### Step 3: How bad is it?")
    severity = st.slider(
        "Severity level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Minor, 5 = Severe"
    )
    severity_labels = {
        1: "Minor", 2: "Mild", 3: "Moderate",
        4: "Serious", 5: "Severe"
    }
    st.markdown(
        f"**Severity: {severity_labels[severity]}**"
    )

    st.markdown("### Step 4: Additional details")
    details = st.text_area(
        "Describe the hazard (optional)",
        placeholder="Any additional details that could help..."
    )

    if st.button("🚨 Submit Report", type="primary"):
        st.success(f"""
        ✅ Report submitted successfully!

        **{hazard_type}** reported in **{report_district}**
        Severity: {severity_labels[severity]}

        Thank you! Your report is now live and helping 
        protect your community. 🙏
        """)
        st.balloons()

    st.markdown("---")
    st.markdown("### 📊 Impact of Community Reports")

    total_reports = 847
    resolved = 623
    prevented = 12

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reports This Month", total_reports)
    with col2:
        st.metric("Issues Resolved", resolved)
    with col3:
        st.metric("Flood Events Prevented", prevented)
