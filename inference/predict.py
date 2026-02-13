import os
import requests
import pandas as pd
import numpy as np
import pymongo
import mlflow
import mlflow.pyfunc
from datetime import datetime
from dotenv import load_dotenv
import dagshub

# -------------------------------
# 1️⃣ CONFIGURATION
# -------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DB_NAME = "aqi_db"
COLLECTION_NAME = "processed_data"

CITY_LAT = 31.5497
CITY_LON = 74.3436
MLFLOW_MODEL_NAME = "AQI_Best_Model"

TRAINING_FEATURES = [
    'clouds', 'day', 'day_of_week', 'hour', 'humidity', 'month', 'pressure', 'temperature', 
    'wind_direction', 'wind_speed', 'aqi_lag_1', 'aqi_lag_6', 'aqi_lag_24', 'aqi_roll_mean_24',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'wind_sin', 'wind_cos'
]

# -------------------------------
# 2️⃣ GLOBAL VARIABLES
# -------------------------------
mongo_client = pymongo.MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

LOADED_MODEL = None
LOADED_MODEL_NAME = "Unknown"

# -------------------------------
# 3️⃣ MLFLOW + DAGSHUB INIT
# -------------------------------
dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)

# -------------------------------
# 4️⃣ LOAD BEST MODEL (by lowest MAE)
# -------------------------------
def load_model_globally():
    """
    Scans ALL versions of registered models to find the absolute best MAE in history.
    """
    global LOADED_MODEL, LOADED_MODEL_NAME

    if LOADED_MODEL is not None:
        return LOADED_MODEL, LOADED_MODEL_NAME

    print("⏳ Scanning history for best model on DagsHub...")

    try:
        client = mlflow.tracking.MlflowClient()
        # Ensure these names match exactly what you used in registry.py
        # If you registered everything under one name like "Lahore_AQI_Model", just use that one string in the list.
        model_names = ["XGBoost", "RandomForest", "RidgeRegression", "NeuralNetwork"] 
        
        best_model_name = None
        best_mae = float("inf")
        best_version_obj = None

        for name in model_names:
            registered_name = name.replace(" ", "_") # e.g. "RandomForest"
            
            try:
                # FIX: Use search_model_versions to get ALL history, not just the latest
                all_versions = client.search_model_versions(f"name='{registered_name}'")
            except:
                continue

            for v in all_versions:
                try:
                    # Get the specific run for this version to check metrics
                    run = client.get_run(v.run_id)
                    mae = float(run.data.metrics.get("mae", float("inf")))
                    
                    # Logic: Keep if MAE is lower. If tie, keep the newer version.
                    if mae < best_mae:
                        best_mae = mae
                        best_model_name = registered_name
                        best_version_obj = v
                    elif mae == best_mae:
                        # Tie-breaker: prefer newer version
                        if best_version_obj and int(v.version) > int(best_version_obj.version):
                            best_version_obj = v
                            best_model_name = registered_name
                except Exception as e:
                    print(f"⚠️ Could not read metrics for {registered_name} v{v.version}: {e}")
                    continue

        if not best_version_obj:
            print("❌ No valid models found in registry.")
            return None, "No Model"

        # Load the winner
        print(f"🏆 Best Model Found: {best_model_name} v{best_version_obj.version} (MAE: {best_mae:.2f})")
        model_uri = f"models:/{best_model_name}/{best_version_obj.version}"
        
        LOADED_MODEL = mlflow.pyfunc.load_model(model_uri)
        LOADED_MODEL_NAME = f"{best_model_name} v{best_version_obj.version}"
        
        return LOADED_MODEL, LOADED_MODEL_NAME

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, "Error"

# -------------------------------
# 5️⃣ WEATHER + PAST AQI DATA
# -------------------------------
def get_weather_forecast():
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={CITY_LAT}&lon={CITY_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") != "200": return []

        forecasts = []
        indices = [0, 8, 16, 24]
        for i in indices:
            if i < len(res['list']):
                item = res['list'][i]
                dt = datetime.fromtimestamp(item['dt'])
                forecasts.append({
                    'dt': dt,
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'wind_deg': item['wind']['deg'],
                    'clouds': item['clouds']['all']
                })
        return forecasts
    except:
        return []

def get_past_aqi(hours_ago):
    target_time = datetime.now().timestamp() - hours_ago * 3600
    doc = collection.find_one(
        {"date": {"$lte": datetime.fromtimestamp(target_time).strftime("%Y-%m-%d %H:%M:%S")}},
        sort=[("date", -1)]
    )
    return doc['aqi'] if doc else 150.0

def get_rolling_mean():
    cursor = collection.find().sort("date", -1).limit(24)
    values = [d['aqi'] for d in cursor]
    return sum(values)/len(values) if values else 150.0

# -------------------------------
# 6️⃣ MAKE PREDICTION
# -------------------------------
def make_prediction():
    model, model_name = load_model_globally()
    if not model:
        return {"error": "Model not found"}

    forecasts = get_weather_forecast()
    if not forecasts:
        return {"error": "Weather Forecast API failed"}

    lag_1 = get_past_aqi(1)
    lag_6 = get_past_aqi(6)
    lag_24 = get_past_aqi(24)
    roll_mean = get_rolling_mean()

    predictions = []

    for f in forecasts:
        data = {
            'temperature': f['temp'],
            'humidity': f['humidity'],
            'pressure': f['pressure'],
            'wind_speed': f['wind_speed'],
            'wind_direction': f['wind_deg'],
            'clouds': f['clouds'],
            'aqi_lag_1': lag_1,
            'aqi_lag_6': lag_6,
            'aqi_lag_24': lag_24,
            'aqi_roll_mean_24': roll_mean,
            'hour': f['dt'].hour,
            'day': f['dt'].day,
            'month': f['dt'].month,
            'day_of_week': f['dt'].weekday(),
            'hour_sin': np.sin(2 * np.pi * f['dt'].hour / 24),
            'hour_cos': np.cos(2 * np.pi * f['dt'].hour / 24),
            'month_sin': np.sin(2 * np.pi * f['dt'].month / 12),
            'month_cos': np.cos(2 * np.pi * f['dt'].month / 12),
            'wind_sin': np.sin(2 * np.pi * f['wind_deg'] / 360),
            'wind_cos': np.cos(2 * np.pi * f['wind_deg'] / 360)
        }

        df = pd.DataFrame([data])
        df = df[TRAINING_FEATURES]

        pred_val = float(model.predict(df)[0])

        if pred_val > 300: desc = "Hazardous"
        elif pred_val > 200: desc = "Very Unhealthy"
        elif pred_val > 150: desc = "Unhealthy"
        elif pred_val > 100: desc = "Sensitive"
        elif pred_val > 50: desc = "Moderate"
        else: desc = "Good"

        predictions.append({
            "date": f['dt'].strftime("%a, %d %b"),
            "aqi": int(pred_val),
            "temp": f['temp'],
            "humidity": f['humidity'],
            "desc": desc
        })

    return {
        "forecast": predictions,
        "model_name": model_name
    }

# -------------------------------
# 7️⃣ PRELOAD MODEL ON START
# -------------------------------
load_model_globally()
