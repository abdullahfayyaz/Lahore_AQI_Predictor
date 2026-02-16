import os
import requests
import pymongo
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

# MongoDB for Cooldown Tracking
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "aqi_db"
COLLECTION_NAME = "alert_logs"  # New collection to store email history

# How many hours to wait before sending another email?
COOLDOWN_HOURS = 6 

def get_last_alert_time():
    """Fetches the timestamp of the last sent email from MongoDB."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Get the most recent log
        last_log = collection.find_one({"_id": "latest_email_alert"})
        
        if last_log:
            return last_log['timestamp']
        return None
    except Exception as e:
        print(f"⚠️ DB Error: {e}")
        return None

def update_last_alert_time():
    """Updates MongoDB with the current time after sending an email."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        collection.update_one(
            {"_id": "latest_email_alert"},
            {"$set": {"timestamp": datetime.now()}},
            upsert=True  # Create if doesn't exist
        )
    except Exception as e:
        print(f"⚠️ Failed to update alert log: {e}")

def send_alert_email(aqi, status, date, temp):
    """Sends the actual email via Brevo."""
    if not API_KEY:
        print("⚠️ Alert skipped: Missing BREVO_API_KEY")
        return

    subject = f"🚨 HAZARDOUS SMOG ALERT: AQI {aqi}"
    
    html_content = f"""
    <html>
    <body>
        <h2 style="color: #7E0023;">⚠️ URGENT AIR QUALITY WARNING</h2>
        <p>The AI Model has detected hazardous smog levels for Lahore.</p>
        <ul>
            <li><strong>Predicted AQI:</strong> <b>{aqi}</b></li>
            <li><strong>Status:</strong> {status}</li>
            <li><strong>Date:</strong> {date}</li>
        </ul>
        <div style="background-color: #f8d7da; padding: 15px; color: #721c24;">
            <strong>Action Required:</strong> Avoid outdoor activities & wear masks.
        </div>
        <p>Stay Safe,<br>Lahore AQI Bot</p>
    </body>
    </html>
    """

    payload = {
        "sender": {"name": "Lahore AQI Bot", "email": SENDER_EMAIL},
        "to": [{"email": RECEIVER_EMAIL}],
        "subject": subject,
        "htmlContent": html_content
    }

    headers = {
        "accept": "application/json",
        "api-key": API_KEY,
        "content-type": "application/json"
    }

    try:
        response = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if response.status_code == 201:
            print(f"✅ 📧 Alert sent successfully to {RECEIVER_EMAIL}")
            # IMPORTANT: Save the time so we don't spam!
            update_last_alert_time()
        else:
            print(f"❌ Failed to send: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def check_and_alert(prediction):
    """
    Checks AQI thresholds AND the cooldown timer.
    """
    aqi = prediction['aqi']
    
    # 1. Check Threshold (e.g., Above 300 for Hazardous)
    ALERT_THRESHOLD = 300 
    
    if aqi <= ALERT_THRESHOLD:
        print(f"✅ AQI {aqi} is safe (No Alert needed).")
        return

    # 2. Check Cooldown (Time Logic)
    last_sent = get_last_alert_time()
    
    if last_sent:
        time_diff = datetime.now() - last_sent
        hours_passed = time_diff.total_seconds() / 3600
        
        if hours_passed < COOLDOWN_HOURS:
            print(f"⏳ Alert SKIPPED: Last email was sent {int(hours_passed)} hours ago (Cooldown: {COOLDOWN_HOURS}h).")
            return

    # 3. Send if passed checks
    print(f"🚨 CRITICAL: AQI {aqi} is Hazardous. Sending Alert...")
    send_alert_email(aqi, prediction['desc'], prediction['date'], prediction['temp'])