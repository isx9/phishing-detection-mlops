# Phishing Detection: an MLOps Workflow

![CI](https://github.com/isx9/phishing-detection-mlops/actions/workflows/ci.yml/badge.svg)

A machine learning pipeline that classifies websites as **phishing** or **legitimate**, built with MLOps practices: a reproducible environment, a preprocessing pipeline, experiment tracking and model registry with MLflow, automated hyperparameter tuning with Optuna, data drift detection with Evidently, automated tests with pytest, continuous integration with GitHub Actions, and model serving through a FastAPI REST API, packaged with Docker and deployed on Kubernetes.

The focus of this project is the engineering workflow around the model: making training reproducible, tracked and tested, then serving the model as a containerized, self-healing service.

---

## Problem and dataset

Phishing websites imitate legitimate ones to steal credentials and personal data. The goal is to detect them automatically from characteristics of their URL and web page.

- **Dataset:** [Web page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) (Kaggle, by Shashwat Tiwari)
- **Size:** 11,430 websites
- **Features:** 87 numeric features describing the URL (length, number of dots, hyphens, use of an IP address...), the page content, and external information (domain age, web traffic, page rank...)
- **Label:** `status`, either `legitimate` or `phishing`
- **Balance:** perfectly balanced, 5,715 websites per class
- **Missing values:** none

## Approach

1. **Preprocessing:** the raw `url` column is dropped, since its useful information is already extracted into the numeric features. The label is encoded as `legitimate = 0`, `phishing = 1`.
2. **Split:** 80% training, 20% test, stratified to keep the 50/50 class ratio, with a fixed random seed for reproducibility.
3. **Model:** a scikit-learn `Pipeline` combining a `SimpleImputer` (fills missing values with the training median) and a Random Forest classifier. Keeping preprocessing inside the model ensures the same steps are applied in training and in production.
4. **Evaluation:** accuracy, precision, recall and F1, computed on the held-out test set.

For a security use case, **recall** is especially important: it measures how many phishing sites are actually caught. A missed phishing site is usually more costly than a false alarm.

## Results

Final model (Random Forest, 100 trees), on the test set:

| Metric | Score |
|---|---|
| Accuracy | 0.960 |
| Precision | 0.957 |
| Recall | 0.964 |
| F1 | 0.960 |

### Experiment: number of trees

Runs tracked and compared with MLflow:

| Trees | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 10 | 0.958 | 0.965 | 0.951 | 0.958 |
| 100 | 0.960 | 0.957 | 0.964 | 0.960 |
| 300 | 0.963 | 0.960 | 0.966 | 0.963 |

Going from 10 to 100 trees improves recall by about 1.3 points. Going from 100 to 300 adds only about 0.2 points, a case of diminishing returns.

### Hyperparameter tuning with Optuna

Optuna searched 20 combinations of `n_estimators` (50–300), `max_depth` (5–30) and `min_samples_split` (2–10), scoring each with 3-fold cross-validation on the training set only, so the test set stays unseen.

- **Best parameters:** 290 trees, `max_depth` 27, `min_samples_split` 2
- **Cross-validation F1:** 0.966
- **Test set:** accuracy 0.962, precision 0.958, recall 0.966, F1 0.962

The tuned model improves F1 by only about 0.2 points over the defaults, and the best values lie close to the default settings (deep trees, minimal splitting). Random Forests perform well out of the box on this dataset. The 100-tree default was kept as the final model: nearly the same performance, with about 3x faster training, which keeps the CI pipeline quick.

## MLOps practices

**Reproducible environment**
All dependencies are pinned to exact versions in `requirements.txt`, so the environment can be recreated on any machine.

**Experiment tracking with MLflow**
Each training run logs its parameters, its metrics, and the trained model with its input signature. Runs can be compared side by side in the MLflow UI. Optuna trials are logged as nested runs under a single parent run.

**Model registry with MLflow**
The trained model is registered as `phishing-detector`, with a new version on each registration. The approved version is marked with the alias `champion`. `predict.py` loads the model by alias (`models:/phishing-detector@champion`), so promoting a new version or rolling back only requires moving the alias, with no code changes.

**Data drift detection with Evidently**
`drift.py` compares new data against the training data (the reference) and generates HTML drift reports with a statistical test per feature. Two scenarios are included:
- test data vs training data: no drift detected (0 of 87 features)
- simulated drift (longer URLs with more dots): drift correctly detected in `length_url` and `nb_dots`

The simulated case also shows a limitation of dataset-level drift flags: with only 2 of 87 features drifted, the overall dataset is not flagged (default threshold: 50% of features), so key features should also be monitored individually.

**Automated tests with pytest**
The tests in `tests/test_model.py` check:
- the data has the expected 87 features and the `url` and `status` columns are removed
- the labels contain only 0 and 1, with no missing values
- the features contain no missing values
- the model returns one valid prediction (0 or 1) per input row
- the model's F1 and recall stay above 0.9
- the pipeline handles inputs with missing values

**Continuous integration with GitHub Actions**
On every push and pull request, the workflow in `.github/workflows/ci.yml`:
1. sets up a clean Linux environment (Ubuntu 24.04) with Python 3.9
2. installs the dependencies from `requirements.txt`
3. lints the code with flake8
4. runs the test suite with pytest

## Model serving

**Exporting the champion model**
`export_model.py` loads the `champion` model from the MLflow registry and saves it to `model/` as a single file, together with `model_info.json` (model name, version and the ordered list of the 87 features). The serving image only contains this exported model, so it does not depend on MLflow or the local `mlruns/` folder. The `model/` folder is generated, so it is not tracked by Git.

**REST API with FastAPI**
`api.py` loads the exported model once at startup and exposes two endpoints:
- `GET /health`: reports that the service is up and which model version it serves
- `POST /predict`: takes the 87 features of a website and returns the predicted label and the phishing probability

Requests are validated automatically against a schema built from the feature list: a request with missing features or non-numeric values is rejected with a `422` error, listing every problem, before it reaches the model. Interactive documentation is generated at `/docs`.

Example request and response:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/phishing.json
```
```json
{"prediction": "phishing", "phishing_probability": 1.0}
```

The `samples/` folder contains one phishing and one legitimate website from the dataset, ready to send (generated by `make_samples.py`).

**Docker image**
The `Dockerfile` uses a multi-stage build:
1. a builder stage installs only the serving dependencies from `requirements-api.txt` (FastAPI, uvicorn, pandas, scikit-learn, joblib, numpy), pinned to the same versions used for training, so the saved model loads in the same environment it was trained in
2. a slim runtime stage copies the installed libraries, `api.py` and the exported model, and starts the API with uvicorn

Dependencies are installed before the application code is copied, so code changes reuse the cached dependency layer and rebuild in seconds. A `.dockerignore` keeps the virtual environment, MLflow runs and Git history out of the build.

**Deployment on Kubernetes**
`k8s.yaml` defines:
- a **Deployment** running 2 replicas of the API, using the local image
- a **readiness probe** on `/health`, so a pod only receives traffic once the model is loaded
- a **liveness probe** on `/health`, so a pod that stops responding is restarted automatically
- a **NodePort Service** that load-balances requests across the replicas and exposes the API on port `30800`

Self-healing was tested by deleting a running pod: Kubernetes immediately created a replacement to restore the desired 2 replicas, which became ready in about 8 seconds, while the other replica kept serving requests.

## Project structure

```
phishing-detection-mlops/
├── .github/workflows/ci.yml   # CI pipeline (GitHub Actions)
├── tests/
│   └── test_model.py          # pytest tests for data, predictions and performance
├── dataset_phishing.csv       # dataset
├── explore.ipynb              # exploratory data analysis
├── train.py                   # data loading, pipeline, training, evaluation, MLflow logging and registration
├── tune.py                    # hyperparameter tuning with Optuna, logged to MLflow
├── predict.py                 # loads the champion model from the MLflow registry and predicts
├── drift.py                   # data drift reports with Evidently
├── export_model.py            # exports the champion model from the registry to model/
├── api.py                     # FastAPI service with /health and /predict
├── make_samples.py            # creates example requests in samples/
├── samples/                   # example requests (one phishing, one legitimate)
├── requirements-api.txt       # pinned dependencies for the serving image
├── Dockerfile                 # multi-stage build of the API image
├── .dockerignore              # files excluded from the Docker build
├── k8s.yaml                   # Kubernetes Deployment and Service
├── pytest.ini                 # pytest configuration
├── requirements.txt           # pinned dependencies for training and development
└── README.md
```

## How to run

Requires Python 3.9.

```bash
# 1. Clone the repository
git clone https://github.com/isx9/phishing-detection-mlops.git
cd phishing-detection-mlops

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Train and register the model (logs the run to MLflow)
python train.py

# 5. Open the MLflow UI at http://127.0.0.1:5001
#    then, under Models > phishing-detector, add the alias "champion" to the version to use
mlflow ui --port 5001

# 6. Predict with the champion model
python predict.py

# 7. Tune hyperparameters with Optuna
python tune.py

# 8. Generate the drift reports
python drift.py

# 9. Run the tests
pytest
```

### Serving the model

Requires Docker. For the Kubernetes step, enable Kubernetes in Docker Desktop (single-node, kubeadm).

```bash
# 1. Export the champion model to model/
python export_model.py

# 2. Run the API locally, then open http://localhost:8000/docs
uvicorn api:app --reload --port 8000

# 3. Build the Docker image and run it
docker build -t phishing-api .
docker run --rm -p 8000:8000 phishing-api

# 4. Send a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @samples/phishing.json

# 5. Deploy on Kubernetes and check the pods
kubectl apply -f k8s.yaml
kubectl get pods

# 6. Send a request through the Kubernetes Service
curl http://localhost:30800/health

# 7. Remove the deployment
kubectl delete -f k8s.yaml
```

## Next steps

- Run drift checks automatically on new incoming data and trigger retraining when key features drift
- Compare additional models (e.g. logistic regression, gradient boosting)
- Add tests for the API endpoints
- Upgrade the project from Python 3.9, which has reached end of life
- Publish the Docker image to a container registry, so the cluster can pull it instead of using a local image

## License

This project is released under the MIT License.