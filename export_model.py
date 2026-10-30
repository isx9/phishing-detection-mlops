import json
import os

import joblib
import mlflow.sklearn
from mlflow import MlflowClient

MODEL_NAME = "phishing-detector"
ALIAS = "champion"
OUTPUT_DIR = "model"


if __name__ == "__main__":
    # load the champion model from the MLflow Model Registry
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version

    # save it as a single file that can be loaded without MLflow
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.joblib"))

    # save some info about it, so the API can report which model it serves
    info = {
        "model_name": MODEL_NAME,
        "version": version,
        "features": list(model.feature_names_in_),
    }
    with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Exported {MODEL_NAME} version {version} to {OUTPUT_DIR}/")
    print(f"The model expects {len(info['features'])} features")