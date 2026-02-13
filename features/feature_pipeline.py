import os
import sys
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import pymongo

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIGURATION ---
MONGO_URI = os.getenv('MONGO_URI')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
PROJECT_NAME = os.getenv('PROJECT_NAME')

# Coordinates (Lahore)
CITY_LAT = 31.5497
CITY_LON = 74.3436

def validate_config():
    """Ensures API keys are set before running."""
    if not MONGO_URI or not OPENWEATHER_API_KEY:
        print("❌ Error: Environment variables are missing.")
        print("Please check your .env file for MONGO_URI and OPENWEATHER_API_KEY.")
        sys.exit(1)

def fetch_data(lat, lon, api_key):
    """
    Fetches BOTH Air Pollution and Weather data.
    """
    try:
        # 1. Fetch Pollution
        poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        poll_res = requests.get(poll_url)
        
        # 2. Fetch Weather
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_res = requests.get(weather_url)

        if poll_res.status_code != 200 or weather_res.status_code != 200:
            print(f"❌ API Error. Poll: {poll_res.status_code}, Weather: {weather_res.status_code}")
            return None, None

        return poll_res.json(), weather_res.json()

    except Exception as e:
        print(f"❌ Network Error: {e}")
        return None, None

def process_data(poll_json, weather_json):
    """
    Combines data into a dictionary (No need for Pandas DataFrame for MongoDB).
    """
    if not poll_json or not weather_json:
        return None

    try:
        # Extract Pollution Data
        poll_list = poll_json['list'][0]
        components = poll_list['components']
        
        # Extract Weather Data
        main = weather_json['main']
        wind = weather_json['wind']
        clouds = weather_json['clouds']
        
        timestamp = poll_list['dt']
        date_obj = datetime.fromtimestamp(timestamp)
        
        # Construct the document
        record = {
            # ID (Primary Key)
            '_id': timestamp,  # Using timestamp as ID ensures no duplicates
            'timestamp': timestamp,
            'date': date_obj.strftime('%Y-%m-%d %H:%M:%S'),
            
            # AQI Data
            'aqi': poll_list['main']['aqi'],
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'no2': components['no2'],
            'so2': components['so2'],
            'co': components['co'],
            'o3': components['o3'],
            
            # Weather Data
            'temperature': main['temp'],
            'humidity': main['humidity'],
            'pressure': main['pressure'],
            'wind_speed': wind['speed'],
            'wind_direction': wind.get('deg', 0),
            'clouds': clouds['all'],
            
            # Time Features
            'hour': date_obj.hour,
            'day': date_obj.day,
            'month': date_obj.month,
            'day_of_week': date_obj.weekday()
        }
        return record
        
    except KeyError as e:
        print(f"❌ Data Parsing Error: Missing key {e}")
        return None

def save_to_mongodb(record, uri, db_name="aqi_db", collection_name="lahore_readings"):
    """
    Saves the record to MongoDB. Uses 'upsert' to prevent duplicates.
    """
    if not record:
        print("⚠️ No record to save.")
        return

    try:
        print("Connecting to MongoDB...")
        client = pymongo.MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # UPSERT: If a document with this _id (timestamp) exists, update it. 
        # If not, insert a new one.
        result = collection.update_one(
            {'_id': record['_id']}, 
            {'$set': record}, 
            upsert=True
        )
        
        if result.upserted_id:
            print("✅ New record inserted successfully!")
        else:
            print("✅ Record updated (Duplicate timestamp handled).")
            
        print(f"📅 Date: {record['date']} | AQI: {record['aqi']} | Temp: {record['temperature']}C")
        
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    validate_config()
    
    print(f"Fetching data for Lat: {CITY_LAT}, Lon: {CITY_LON}...")
    
    poll_data, weather_data = fetch_data(CITY_LAT, CITY_LON, OPENWEATHER_API_KEY)
    
    if poll_data and weather_data:
        record = process_data(poll_data, weather_data)
        save_to_mongodb(record, MONGO_URI)
    else:
        print("❌ Failed to fetch valid data.")