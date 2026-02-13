import os
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "aqi_db"
SOURCE_COLLECTION = "lahore_readings"
TARGET_COLLECTION = "processed_data"

def calculate_aqi_us(pm25, pm10):
    """
    Calculates US EPA AQI based on PM2.5 and PM10 concentrations.
    """
    def calc_pollutant_aqi(conc, breakpoints):
        """Helper to find the AQI bucket."""
        for (c_low, c_high, i_low, i_high) in breakpoints:
            if c_low <= conc <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
        return 500 # Cap at 500 if insanely high

    # PM2.5 Breakpoints (US EPA Standard)
    pm25_breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]

    # PM10 Breakpoints
    pm10_breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]

    aqi_25 = calc_pollutant_aqi(pm25, pm25_breakpoints)
    aqi_10 = calc_pollutant_aqi(pm10, pm10_breakpoints)
    
    # The final AQI is the MAXIMUM of the individual pollutant AQIs
    return max(aqi_25, aqi_10)

def load_data():
    print(f"⏳ Loading raw data from '{SOURCE_COLLECTION}'...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[SOURCE_COLLECTION]
    
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    # Basic cleaning
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if '_id' in df.columns: df = df.drop(columns=['_id'])
    if 'timestamp' in df.columns: df = df.drop(columns=['timestamp'])
    
    print(f"✅ Loaded {len(df)} records.")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")
    df = df.set_index('date')
    df = df[~df.index.duplicated(keep='first')]
    df = df.interpolate(method='time')
    df = df.bfill().ffill()
    df = df.reset_index()
    return df

def feature_engineering(df):
    print("⚙️ Engineering features...")
    
    # --- 1. CALCULATE REAL AQI (0-500) ---
    print("   🧮 Converting Raw PM2.5/PM10 to US AQI Scale...")
    
    # Apply the function to every row
    df['calculated_aqi'] = df.apply(
        lambda row: calculate_aqi_us(row['pm2_5'], row['pm10']), axis=1
    )
    
    # OVERWRITE the simple 'aqi' column with our new Calculation
    df['aqi'] = df['calculated_aqi']
    df = df.drop(columns=['calculated_aqi'])

    # --- 2. LAG FEATURES ---
    target_col = 'aqi'
    df['aqi_lag_1'] = df[target_col].shift(1)
    df['aqi_lag_6'] = df[target_col].shift(6)
    df['aqi_lag_24'] = df[target_col].shift(24)
    df['aqi_roll_mean_24'] = df[target_col].rolling(window=24).mean()
    
    # --- 3. CYCLICAL FEATURES ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'wind_direction' in df.columns:
        df['wind_sin'] = np.sin(2 * np.pi * df['wind_direction'] / 360)
        df['wind_cos'] = np.cos(2 * np.pi * df['wind_direction'] / 360)

    df = df.dropna()
    return df

def save_to_mongo(df):
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[TARGET_COLLECTION]
    
    collection.delete_many({})
    
    df_save = df.copy()
    df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    records = df_save.to_dict("records")
    
    if records:
        collection.insert_many(records)
        print(f"💾 Saved {len(records)} records (with calculated AQI) to '{TARGET_COLLECTION}'.")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = clean_data(df)
        df = feature_engineering(df)
        save_to_mongo(df)
        print("\n🎉 Sample Calculated AQI:")
        print(df[['date', 'pm2_5', 'pm10', 'aqi']].head())