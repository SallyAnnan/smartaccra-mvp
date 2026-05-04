#  SmartAccra — Urban Risk Intelligence Platform

> **Know before it happens.**

SmartAccra is a data-driven urban risk intelligence platform designed 
for residents of Accra, Ghana. It provides real-time predictions and 
community-powered alerts for the three most critical urban risks 
facing Accra's informal settlements — flooding, fire hazards, and 
waste management failures.

## SDG Alignment
This project directly addresses **SDG 11 — Sustainable Cities and 
Communities**, specifically targets 11.5 (disaster risk reduction) 
and 11.6 (environmental impact and waste management).

##  The Problem
Accra's informal settlements face three recurring preventable crises:
- **Flooding** caused by blocked drainage systems during rainy season
- **Fires** spreading through densely packed buildings with illegal wiring
- **Waste accumulation** that blocks gutters and directly triggers floods

Residents currently have **no warning system** and **no data tools** 
to anticipate or prepare for these disasters.

##  The Solution
SmartAccra combines three data modules into one platform:

| Module | Data Source | Model Type |
|--------|------------|------------|
| 🌊 Flood Risk | Open-Meteo Weather API (real-time) | Time Series Forecasting |
| 🔥 Fire Risk | OpenStreetMap building density data | Logistic Regression Classification |
| 🗑️ Waste & Drainage | Accra district data + community reports | Descriptive Analytics |

## 🗺️ Target Users
Residents of Accra's informal settlements including Adenta, Medina, 
Nima, Ablekuma, Dansoman, Mamprobi and surrounding areas.

## 🛠️ Tech Stack
- **Python** — data processing and machine learning
- **Streamlit** — web application framework
- **Folium** — interactive mapping with real OpenStreetMap data
- **Open-Meteo API** — live weather and rainfall forecasts for Accra
- **Scikit-learn** — classification and regression models
- **Pandas & Plotly** — data manipulation and visualisation

## Data Sources
- [Open-Meteo](https://open-meteo.com/) — Free real-time weather API
- [OpenStreetMap Overpass API](https://overpass-api.de/) — Building 
  density and infrastructure data for Accra
- [World Bank Urban Data](https://data.worldbank.org/) — Accra 
  district population and urbanisation data
- Community-reported hazard data (simulated for MVP)

## Live Demo
[Click here to view the live app](#) ← *link added after deployment*

## Developer
Built by Sally Annan as part of a data products course project, 
addressing real urban challenges in Accra, Ghana.
