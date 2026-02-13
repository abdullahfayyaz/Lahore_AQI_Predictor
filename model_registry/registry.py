import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from dotenv import load_dotenv
import dagshub

# Load environment variables
load_dotenv()

# Initialize DagsHub MLflow
# dagshub.init(
#     repo_owner=os.getenv("DAGSHUB_USERNAME"),
#     repo_name=os.getenv("DAGSHUB_REPO"),
#     mlflow=True
# )
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# --- AUTHENTICATION CHECK ---
if not DAGSHUB_TOKEN or not DAGSHUB_USERNAME:
    raise EnvironmentError("❌ Critical Error: DAGSHUB_TOKEN or DAGSHUB_USERNAME is missing. Check your GitHub Secrets!")

# --- CONNECT SILENTLY (No Browser) ---
print(f"🔌 Connecting to DagsHub: {DAGSHUB_REPO}...")
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Set credentials explicitly so MLflow doesn't ask for them
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

MLFLOW_EXP_NAME = "Lahore_AQI_Experiment"
mlflow.set_experiment(MLFLOW_EXP_NAME)

def log_model_run(model_name, model, params, metrics):
    """
    Logs a single training run to MLflow and registers it.
    Each model type gets its own registered model.
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log params and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Use the model_name as the registered model name
        registered_model_name = model_name.replace(" ", "_")  # Replace spaces for MLflow

        # Log and register model
        if "XGBoost" in model_name:
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name
            )

        print(f"✅ Logged {model_name} | MAE: {metrics['mae']:.2f}")

        # Auto-promote to Production if this run has the best MAE for this model
        # client = mlflow.tracking.MlflowClient()
        # try:
        #     # Get all versions of this registered model
        #     versions = client.get_latest_versions(registered_model_name, stages=["Production", "None"])
        #     if versions:
        #         # Find current production MAE (if any)
        #         prod_version = next((v for v in versions if v.current_stage=="Production"), None)
        #         if prod_version is None or metrics['mae'] < float(prod_version.run_id):
        #             # Promote this run to Production
        #             client.transition_model_version_stage(
        #                 name=registered_model_name,
        #                 version=run.info.run_id,  # MLflow run_id is used as version in DagsHub
        #                 stage="Production",
        #                 archive_existing_versions=True
        #             )
        #             print(f"🚀 {model_name} promoted to Production!")
        # except Exception as e:
        #     print(f"⚠️ Could not auto-promote: {e}")