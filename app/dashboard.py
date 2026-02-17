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
    # 1. Good (0-50)
    if aqi <= 50:
        recs.append("✅ Air quality is satisfactory")
        recs.append("🏃‍♂️ Enjoy your usual outdoor activities")
        recs.append("🏠 Open windows to bring in fresh air")

    # 2. Moderate (51-100)
    elif aqi <= 100:
        recs.append("⚠️ Sensitive groups should reduce outdoor exercise")
        recs.append("😷 Sensitive groups: Wear a mask if experiencing symptoms")
        recs.append("🏠 Close windows to avoid outdoor pollution")
        recs.append("🏃‍♂️ General public: Enjoy outdoor activities")

    # 3. Unhealthy for Sensitive Groups (101-150)
    elif aqi <= 150:
        recs.append("🏃‍♂️ Reduce outdoor exercise")
        recs.append("🏠 Close your windows to avoid dirty outdoor air")
        recs.append("😷 Sensitive groups should wear a mask outdoors")
        recs.append("💨 Run an air purifier")

    # 4. Unhealthy (151-200)
    elif aqi <= 200:
        recs.append("⛔ Avoid prolonged outdoor exertion")
        recs.append("😷 Wear a mask outdoors")
        recs.append("🏃‍♂️ Move exercise indoors")
        recs.append("💨 Run air purifier on High")
        recs.append("🏠 Keep windows and doors closed")

    # 5. Very Unhealthy (201-300)
    elif aqi <= 300:
        recs.append("⛔ Avoid all outdoor exertion")
        recs.append("😷 Wear a mask if you must go out")
        recs.append("👶 Children & Elderly should stay indoors")
        recs.append("💨 Run air purifier on Max")

    # 6. Hazardous (301+)
    else:
        recs.append("☠️ EMERGENCY CONDITIONS: Stay indoors")
        recs.append("⛔ Avoid all physical activity outdoors")
        recs.append("😷 Wear an N95/P100 mask if strictly necessary")
        recs.append("💨 Seal windows/doors and run air purifier")
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
    # 1. Header & Refresh Button Layout
    col_head, col_btn = st.columns([6, 1])
    
    with col_head:
        st.subheader("Current Air Quality Status")
        
    with col_btn:
        # pressing this button triggers a script rerun, fetching new data automatically
        if st.button("🔄 Refresh", help="Fetch latest prediction from API"):
            st.rerun()

    # 2. Fetch Data
    with st.spinner('Fetching latest AI prediction...'):
        data = get_live_prediction()
    
    if data and "forecast" in data:
        current = data["forecast"][0]
        forecasts = data["forecast"][1:]
        
        # 3. Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        status, color, icon = get_aqi_status(current['aqi'])
        
        with col1:
            st.metric("Current AQI", f"{current['aqi']}", delta_color="inverse")
        with col2:
            st.metric("Temperature", f"{current['temp']}°C")
        with col3:
            st.metric("Humidity", f"{current['humidity']}%")
        with col4:
            st.markdown(f"""
                <div style='background-color:{color}; color:white; padding:10px; 
                border-radius:5px; text-align:center; font-weight:bold; 
                margin-top: 5px; box-shadow: 0px 2px 5px rgba(0,0,0,0.2);'>
                {icon} {status}
                </div>
                """, unsafe_allow_html=True)

        # 4. Gauge & Recommendations
        st.divider()
        c_gauge, c_recs = st.columns([1.2, 1]) # Gauge gets slightly more space
        
        with c_gauge:
            st.plotly_chart(create_gauge(current['aqi']), use_container_width=True)
            
        with c_recs:
            st.subheader("💡 Health Actions")
            for rec in get_recommendations(current['aqi']):
                st.info(rec)

        # 5. Forecast Cards
        st.divider()
        st.subheader("📅 3-Day Forecast")
        
        # Check if we have forecasts
        if forecasts:
            f_cols = st.columns(3)
            for i, f in enumerate(forecasts[:3]):
                f_status, f_color, f_icon = get_aqi_status(f['aqi'])
                with f_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 5px solid {f_color};">
                        <h4>{f['date']}</h4>
                        <h1 style="color:{f_color}; font-size: 3rem; margin:0;">{f['aqi']}</h1>
                        <p style="font-weight:bold; font-size: 1.1rem;">{f_icon} {f_status}</p>
                        <hr style="margin: 10px 0;">
                        <p>🌡️ {f['temp']}°C &nbsp;|&nbsp; 💧 {f['humidity']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient forecast data available.")
                
    else:
        st.error("⚠️ API is offline or returning empty data. Please check your backend.")
        if st.button("Retry Connection"):
            st.rerun()

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

        # 1. Filter Data 
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= "2025-08-01") & (df['date'] <= "2026-01-31")
        filtered_df = df.loc[mask].copy()
        
        if not filtered_df.empty:
            # 2. ALWAYS Apply Smoothing (24h Average)
            # Set index to date for resampling
            filtered_df = filtered_df.set_index('date')
            
            # Resample to hourly to ensure continuity (ignoring non-numeric cols like _id)
            filtered_df = filtered_df.resample('h').mean(numeric_only=True).interpolate()
            
            # Calculate rolling average
            filtered_df['aqi'] = filtered_df['aqi'].rolling(window=24, min_periods=1).mean()
            
            # Reset index so 'date' becomes a column again for Plotly
            filtered_df = filtered_df.reset_index()

            # 3. Create the Clean Chart
            fig_trend = px.line(
                filtered_df, 
                x="date", 
                y="aqi", 
                title="AQI Trend Analysis",
                color_discrete_sequence=["#FF4B4B"], 
                labels={"aqi": "AQI Level", "date": "Date"}
            )
            
            # 4. Polish the Layout
            fig_trend.update_layout(
                xaxis_title=None, 
                yaxis_title="AQI (24h Avg)",
                hovermode="x unified",
                height=400,
                xaxis=dict(
                    rangeslider=dict(visible=True), 
                    type="date"
                )
            )
            
            # Clean gridlines
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor='lightgray')

            st.plotly_chart(fig_trend, use_container_width=True)
            
        else:
            st.info("ℹ️ No data available for the period Aug 2025 - Jan 2026.")
        
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
    - **Models:** RidgeRegression, RandomForestRegressor, XGBRegressor, NeuralNetwork
    - **Tracking:** MLflow & DagsHub
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Database:** MongoDB Atlas
    
    **Developer:** Abdullah Fayyaz
    """)