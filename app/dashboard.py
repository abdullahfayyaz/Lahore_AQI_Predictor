import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
import pandas as pd
import pymongo
import os
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="Lahore AQI Monitor", page_icon="😷", layout="wide")
load_dotenv()

API_URL = "http://127.0.0.1:8000/predict"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "aqi_db"
COLLECTION_NAME = "processed_data"

# --- CUSTOM CSS FOR "GLASS" CARDS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #486d8a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        color: black;
    }
    .metric-card h4 {
        color: black !important;
    }
    .stMetric { text-align: center !important; }
    </style>
""", unsafe_allow_html=True)

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=600)  # Cache for 10 minutes to prevent database overload
def fetch_historical_data():
    """Fetches historical data from MongoDB for EDA."""
    if not MONGO_URI: return pd.DataFrame()
    
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Fetch last 1000 records for speed
        cursor = collection.find().sort("date", -1).limit(1000)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()

def get_live_prediction():
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# --- HEALTH LOGIC ---
def get_aqi_status(aqi):
    if aqi <= 50: return "Good", "green", "😊"
    elif aqi <= 100: return "Moderate", "#FFDE33", "😐"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "orange", "😷"
    elif aqi <= 200: return "Unhealthy", "red", "🤢"
    elif aqi <= 300: return "Very Unhealthy", "purple", "☠️"
    else: return "Hazardous", "#7E0023", "🧟"

def get_recommendations(aqi):
    recs = []
    if aqi <= 100:
        recs.append("✅ It's a great day for outdoor sports!")
        recs.append("🏠 Open windows to let fresh air in.")
    elif aqi <= 150:
        recs.append("⚠️ Sensitive groups should wear masks.")
        recs.append("🏃‍♂️ Reduce prolonged outdoor exertion.")
    else:
        recs.append("⛔ Avoid ALL outdoor exercise.")
        recs.append("😷 Wear an N95 mask if you must go out.")
        recs.append("💨 Run your air purifier on High.")
    return recs

# --- VISUALIZATION FUNCTIONS ---
def create_gauge(aqi):
    status, color, icon = get_aqi_status(aqi)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<span style='font-size:20px'>{icon} {status}</span>"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "#FFDE33"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
                {'range': [300, 500], 'color': "#7E0023"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': aqi}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- MAIN LAYOUT ---
st.title("🏭 Lahore AQI Monitor")
st.markdown("Real-time forecasts")

# Tabs for Organization
tab_live, tab_analysis, tab_about = st.tabs(["📡 Live Dashboard", "📊 Deep Analysis (EDA)", "ℹ️ About"])

# =========================================
# TAB 1: LIVE DASHBOARD
# =========================================
with tab_live:
    data = get_live_prediction()
    
    if data and "forecast" in data:
        current = data["forecast"][0]
        forecasts = data["forecast"][1:]
        
        # 1. Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        status, color, icon = get_aqi_status(current['aqi'])
        
        with col1:
            st.metric("Current AQI", f"{current['aqi']}", delta_color="inverse")
        with col2:
            st.metric("Temperature", f"{current['temp']}°C")
        with col3:
            st.metric("Humidity", f"{current['humidity']}%")
        with col4:
            st.markdown(f"<div style='background-color:{color}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>{status}</div>", unsafe_allow_html=True)

        # 2. Gauge & Recommendations
        c_gauge, c_recs = st.columns([1, 1])
        with c_gauge:
            st.plotly_chart(create_gauge(current['aqi']), use_container_width=True)
        with c_recs:
            st.subheader("💡 Health Actions")
            for rec in get_recommendations(current['aqi']):
                st.info(rec)

        # 3. Forecast Cards
        st.divider()
        st.subheader("📅 3-Day Forecast")
        f_cols = st.columns(3)
        for i, f in enumerate(forecasts[:3]):
            f_status, f_color, f_icon = get_aqi_status(f['aqi'])
            with f_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{f['date']}</h4>
                    <h1 style="color:{f_color}">{f['aqi']}</h1>
                    <p>{f_icon} {f_status}</p>
                    <hr>
                    <p>🌡️ {f['temp']}°C | 💧 {f['humidity']}%</p>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.error("⚠️ API is offline. Please run `uvicorn app.main:app --reload`.")

# =========================================
# TAB 2: DEEP ANALYSIS (EDA)
# =========================================
with tab_analysis:
    st.header("🔍 Historical Data Analysis")
    df = fetch_historical_data()
    
    if not df.empty:
        # 1. Hourly Trends (Seasonality)
        st.subheader("🕒 When is Smog Worst? (Hourly Analysis)")
        st.markdown("This box plot shows the spread of AQI for each hour of the day based on historical data.")
        
        fig_hourly = px.box(df, x="hour", y="aqi", color="hour", 
                            title="AQI Distribution by Hour of Day",
                            labels={"hour": "Hour (24h)", "aqi": "Air Quality Index"})
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.caption("💡 **Insight:** AQI usually spikes late at night and early morning due to temperature inversion.")

        # 2. Correlation Heatmap
        st.divider()
        st.subheader("🔥 What Drives Smog? (Correlation Matrix)")
        
        corr_cols = ['aqi', 'temperature', 'humidity', 'wind_speed', 'pressure']
        corr_matrix = df[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("💡 **Insight:** Red = Strong Positive Correlation (Both go up). Blue = Negative Correlation (One goes up, other goes down).")

        # 3. Trend Over Time
        st.divider()
        st.subheader("📈 Long-Term Trend")
        fig_trend = px.line(df, x="date", y="aqi", title="AQI History Over Time", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.warning("No historical data found in MongoDB. Run your data pipeline to populate analytics.")

# =========================================
# TAB 3: ABOUT
# =========================================
with tab_about:
    st.markdown("""
    ### About this Project
    This AI-powered dashboard predicts Air Quality in Lahore using advanced machine learning.
    
    **Tech Stack:**
    - **Model:** XGBoost (Regression)
    - **Tracking:** MLflow & DagsHub
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Database:** MongoDB Atlas
    
    **Developer:** Abdullah Fayyaz
    """)