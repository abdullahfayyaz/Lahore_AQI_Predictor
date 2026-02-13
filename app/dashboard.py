import streamlit as st
import plotly.graph_objects as go
import requests

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"

# --- 1. HEALTH RECOMMENDATIONS FUNCTION ---
def get_health_recommendations(aqi):
    if aqi is None: aqi = 0
    recommendations = []
    
    if aqi <= 50:
        status = "Good"
        color = "green"
        recommendations.append("Open your windows to bring in fresh air")
        recommendations.append("Enjoy outdoor activities")
    elif aqi <= 100:
        status = "Moderate"
        color = "#FFDE33" # Yellow
        recommendations.append("Sensitive groups should reduce outdoor exercise")
        recommendations.append("Close your windows to avoid dirty outdoor air")
        recommendations.append("Get a monitor to track air quality")
    elif aqi <= 150:
        status = "Unhealthy for Sensitive Groups"
        color = "orange"
        recommendations.append("Sensitive groups should wear a mask outdoors")
        recommendations.append("Get an air purifier if you don't have one")
        recommendations.append("Run an air purifier")
        recommendations.append("Close your windows to avoid dirty outdoor air")
        recommendations.append("Reduce outdoor exercise")
    elif aqi <= 200:
        status = "Unhealthy"
        color = "red"
        recommendations.append("Wear a mask outdoors")
        recommendations.append("Run an air purifier")
        recommendations.append("Close your windows to avoid dirty outdoor air")
        recommendations.append("Avoid outdoor exercise")
        recommendations.append("Get a monitor to track indoor air")
    elif aqi <= 300:
        status = "Very Unhealthy"
        color = "purple"
        recommendations.append("Wear a mask outdoors")
        recommendations.append("Run an air purifier on high")
        recommendations.append("Avoid outdoor exercise")
        recommendations.append("Close your windows to avoid dirty outdoor air")
    else:
        status = "Hazardous"
        color = "#7E0023" # Maroon
        recommendations.append("Wear a mask outdoors (N95)")
        recommendations.append("Run an air purifier")
        recommendations.append("Avoid all outdoor exercise")
        recommendations.append("Close your windows")

    return status, color, recommendations

# --- 2. GAUGE CHART FUNCTION ---
def create_aqi_gauge(aqi_value):
    status, color, _ = get_health_recommendations(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>Current AQI Status: {status}</b>", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "black", 'thickness': 0.05}, 
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "#FFDE33"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
                {'range': [300, 500], 'color': "#7E0023"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- MAIN APP DISPLAY ---
st.title("Air Quality Dashboard")

# Initialize variables
current_aqi = 0
current_temp = 0
current_humidity = 0
forecast_list = []
api_success = False

# --- FETCH DATA FROM API ---
with st.spinner('Fetching latest predictions...'):
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()
            full_forecast = data.get("forecast", [])
            
            if full_forecast and len(full_forecast) > 0:
                # 1. Current Data (Index 0)
                current_data = full_forecast[0]
                current_aqi = current_data.get('aqi', 0)
                current_temp = current_data.get('temp', 0)
                current_humidity = current_data.get('humidity', 0) # <--- GETTING HUMIDITY
                
                # 2. Future Forecast (Index 1+)
                if len(full_forecast) > 1:
                    forecast_list = full_forecast[1:]
                
                api_success = True
        else:
            st.error(f"API Error: Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to API at {API_URL}")

if api_success:
    # 1. DISPLAY GAUGE & METRICS
    st.subheader("Today's Forecast")
    
    # Create 3 columns for metrics below gauge
    st.plotly_chart(create_aqi_gauge(current_aqi), use_container_width=True)
    
    m1, m2 = st.columns(2)
    m1.metric(label="Temperature", value=f"{current_temp} °C")
    m2.metric(label="Humidity", value=f"{current_humidity}%") # <--- DISPLAYING HUMIDITY
    
    st.divider()

    # 2. HEALTH RECOMMENDATIONS
    _, _, recs = get_health_recommendations(current_aqi)
    st.subheader("Health Recommendations")
    
    rec_cols = st.columns(2)
    for i, rec in enumerate(recs):
        with rec_cols[i % 2]:
            st.info(rec)
    st.divider()

    # 3. DISPLAY 3-DAY FORECAST
    if forecast_list:
        st.subheader("Next 3-Day Forecast")
        cols = st.columns(min(len(forecast_list), 3))

        for i, col in enumerate(cols):
            day_data = forecast_list[i]
            
            d_aqi = day_data.get('aqi', 0)
            d_temp = day_data.get('temp', 0)
            d_hum = day_data.get('humidity', 0) # <--- GETTING FORECAST HUMIDITY
            d_date = day_data.get('date', 'Unknown')
            
            status, color, _ = get_health_recommendations(d_aqi)
            
            with col:
                st.markdown(f"### {d_date}")
                st.markdown(f"**AQI:** <span style='color:{color}; font-size:24px'><b>{d_aqi}</b></span>", unsafe_allow_html=True)
                st.markdown(f"*{status}*")
                st.markdown(f"🌡️ **Temp:** {d_temp} °C")
                st.markdown(f"💧 **Humidity:** {d_hum}%") # <--- DISPLAYING IT
else:
    st.warning("Waiting for API connection...")

# Run with: streamlit run app/dashboard.py            