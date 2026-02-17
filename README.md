# 🏭 Lahore AQI Monitor & Forecaster

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

An end-to-end Machine Learning application that monitors, analyzes, and forecasts the Air Quality Index (AQI) for **Lahore, Pakistan**. It uses real-time weather data and historical pollution trends to predict smog levels up to 3 days in advance, providing health recommendations and hazardous alerts.

---

## 🌟 Key Features

* **📡 Real-Time Dashboard:** Interactive Streamlit UI displaying live AQI, Temperature, and Humidity.
* **📅 3-Day Forecast:** Machine Learning-powered predictions for the next 72 hours.
* **🚨 Smart Alerts:** Automated email notifications (via Brevo) when AQI hits **Hazardous (300+)** levels (includes a 6-hour cooldown to prevent spam).
* **📊 Deep Analysis (EDA):** Built-in visualization of historical trends, seasonality (hourly patterns), and feature correlations.
* **🧪 MLOps Integration:** Full experiment tracking and model registry using **MLflow** & **DagsHub**.
* **🧠 Advanced Modeling:** Uses **XGBoost** and **Neural Networks** with lag features and rolling means for high accuracy.

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit, Plotly
* **Backend:** FastAPI, Python
* **Database:** MongoDB Atlas (NoSQL)
* **ML Engine:** Scikit-Learn, XGBoost, SHAP
* **MLOps:** MLflow, DagsHub
* **APIs:** OpenWeatherMap (Weather), Brevo (Email Alerts)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdullahfayyaz/Lahore_AQI_Predictor.git
cd Lahore_AQI_Predictor
```
 
### 2. Create a Virtual Environment (Recommended)
Isolate your project dependencies to avoid conflicts.

Windows: 
```bash
python -m venv venv
```
Mac/Linux: 
```bash
python3 -m venv venv
```

Activate it:

Windows (Command Prompt):
```bash
venv\Scripts\activate
```
Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

Mac / Linux:
```bash
source venv/bin/activate
```

(You will know it worked when you see (venv) appear at the start of your terminal line.)

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a .env file in the root directory and add your credentials:
```ini
# --- Database ---
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.mongodb.net/

# --- APIs ---
OPENWEATHER_API_KEY=your_openweather_api_key
BREVO_API_KEY=your_brevo_api_key_for_emails

# --- Email Config ---
SENDER_EMAIL=your_email@gmail.com
RECEIVER_EMAIL=your_email@gmail.com

# --- MLOps (DagsHub) ---
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO=your_repo_name
DAGSHUB_TOKEN=your_dagshub_token
```


## 🚀 How to Run

### 1. Start the API Backend
This serves the ML model predictions via FastAPI.
```bash
uvicorn app.api:app --reload
```

### 2. Launch the Dashboard
Open a new terminal (activate venv there too if not activate) and run the Streamlit app.
```bash
streamlit run app/dashboard.py
```
The dashboard will automatically open at http://localhost:8501