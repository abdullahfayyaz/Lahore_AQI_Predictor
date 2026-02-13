import os
import requests
import time
import pymongo
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MONGO_URI = os.getenv('MONGO_URI')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
CITY_LAT = 31.5497
CITY_LON = 74.3436

# --- DATES (6 Months Back) ---
START_DATE = "2025-07-24"  # <--- Updated for 6 months
END_DATE = "2026-01-24"

# --- SAFETY BRAKE ---
MAX_DAILY_CALLS = 1000

def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client["aqi_db"]["lahore_readings"]

def get_unix_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

# --- STAGE 1: POLLUTION HISTORY ---
def fetch_pollution_history(start_unix, end_unix):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": CITY_LAT, "lon": CITY_LON,
        "start": start_unix, "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }
    try:
        res = requests.get(url, params=params)
        return res.json().get('list', []) if res.status_code == 200 else []
    except Exception as e:
        print(f"❌ Pollution Request Failed: {e}")
        return []

def save_pollution_to_mongo(history_list):
    collection = get_mongo_collection()
    operations = []
    
    for item in history_list:
        timestamp = item['dt']
        date_obj = datetime.fromtimestamp(timestamp)
        
        # Use $setOnInsert for weather fields to avoid overwriting existing data
        update_doc = {
            '$set': {
                'aqi': item['main']['aqi'],
                'pm2_5': item['components']['pm2_5'],
                'pm10': item['components']['pm10'],
                'no2': item['components']['no2'],
                'so2': item['components']['so2'],
                'co': item['components']['co'],
                'o3': item['components']['o3'],
                'hour': date_obj.hour,
                'day': date_obj.day,
                'month': date_obj.month,
                'day_of_week': date_obj.weekday(),
                'date': date_obj.strftime('%Y-%m-%d %H:%M:%S')
            },
            '$setOnInsert': {
                'temperature': None, 'humidity': None, 'pressure': None,
                'wind_speed': None, 'wind_direction': None, 'clouds': None
            }
        }
        operations.append(pymongo.UpdateOne({'_id': timestamp}, update_doc, upsert=True))

    if operations:
        collection.bulk_write(operations)
        print(f"✅ Saved/Updated {len(operations)} pollution records.")

# --- STAGE 2: WEATHER HISTORY ---
def fetch_weather_timemachine(timestamp):
    url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        "lat": CITY_LAT, "lon": CITY_LON,
        "dt": timestamp,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            data = res.json().get('data', [])
            return data[0] if data else None
        else:
            print(f"❌ Weather API Error {res.status_code}: {res.text}")
            return None
    except Exception as e:
        print(f"❌ Weather Request Failed: {e}")
        return None

def enrich_data_with_weather():
    collection = get_mongo_collection()
    
    # Only find records missing temperature
    query = {"temperature": None}
    missing_count = collection.count_documents(query)
    
    if missing_count == 0:
        print("🎉 All records have weather data! Ready to train.")
        return

    print(f"📉 Found {missing_count} records missing weather data.")
    print(f"⚡ Starting enrichment (Max {MAX_DAILY_CALLS} calls)...")

    # Sort by timestamp ascending (fill July/August first)
    cursor = collection.find(query).sort("timestamp", 1).limit(MAX_DAILY_CALLS)
    
    calls_made = 0
    
    for doc in cursor:
        timestamp = doc['_id']
        weather = fetch_weather_timemachine(timestamp)
        
        if weather:
            collection.update_one(
                {'_id': timestamp},
                {'$set': {
                    'temperature': weather.get('temp'),
                    'humidity': weather.get('humidity'),
                    'pressure': weather.get('pressure'),
                    'wind_speed': weather.get('wind_speed'),
                    'wind_direction': weather.get('wind_deg'),
                    'clouds': weather.get('clouds')
                }}
            )
            print(f"✅ Enriched {doc['date']} | Temp: {weather.get('temp')}C")
            calls_made += 1
        
        time.sleep(0.2) 

    print(f"🛑 Daily limit reached. Total calls made: {calls_made}")
    if missing_count > calls_made:
        print(f"👉 {missing_count - calls_made} records remaining. Run this script again tomorrow!")

# --- MAIN RUNNER ---
def run_pipeline():
    print("--- Stage 1: Backfilling Pollution Data (Free) ---")
    current_start = get_unix_timestamp(START_DATE)
    final_end = get_unix_timestamp(END_DATE)
    
    # 30-day chunks for pollution history API
    CHUNK_SIZE = 30 * 24 * 60 * 60 
    
    while current_start < final_end:
        current_end = min(current_start + CHUNK_SIZE, final_end)
        print(f"Fetching pollution from {datetime.fromtimestamp(current_start).date()}...")
        data = fetch_pollution_history(current_start, current_end)
        if data:
            save_pollution_to_mongo(data)
        current_start = current_end + 1
    
    print("\n--- Stage 2: Enriching with Weather History (Quota Limited) ---")
    enrich_data_with_weather()

if __name__ == "__main__":
    if not MONGO_URI or not OPENWEATHER_API_KEY:
        print("❌ Error: Missing .env configuration.")
    else:
        run_pipeline()