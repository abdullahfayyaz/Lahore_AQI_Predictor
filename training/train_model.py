import os, sys
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.evaluate import calculate_metrics
from model_registry.registry import log_model_run

load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "aqi_db"
COLLECTION_NAME = "processed_data"
DEPLOY_MODE = False  # True = train on full dataset

def load_data():
    client = pymongo.MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    df = pd.DataFrame(list(collection.find()))
    
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
        drop_cols = ['date','_id','pm2_5','pm10','no2','so2','co','o3','calculated_aqi']
        df = df.drop(columns=drop_cols, errors='ignore')
    return df

def main():
    df = load_data()
    if df.empty: return print("❌ No data found.")

    target = 'aqi'
    X = df.drop(columns=[target])
    y = df[target]

    if DEPLOY_MODE:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42), {"n_estimators":100}),
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42), {"lr":0.1}),
        ("RidgeRegression", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), {"alpha":1.0}),
        ("NeuralNetwork", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)), {"layers":"(64,32)","max_iter":1000})
    ]

    best_mae = float("inf")
    best_name = ""
    print("\n🚀 Starting Training Loop...")

    for name, model, params in models:
        print(f"🏃 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = calculate_metrics(y_test, preds)
        log_model_run(name, model, params, metrics)

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name

    print(f"\n🏆 WINNER: {best_name} (MAE: {best_mae:.2f})")

if __name__ == "__main__":
    main()
