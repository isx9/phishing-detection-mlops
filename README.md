# Phishing Detection: an MLOps Workflow

![CI](https://github.com/isx9/phishing-detection-mlops/actions/workflows/ci.yml/badge.svg)

A machine learning pipeline that classifies websites as **phishing** or **legitimate**, built with MLOps practices: a reproducible environment, a preprocessing pipeline, experiment tracking and model registry with MLflow, automated hyperparameter tuning with Optuna, data drift detection with Evidently, automated tests with pytest, and continuous integration with GitHub Actions.

The focus of this project is the engineering workflow around the model: making training reproducible, tracked, tested, and ready to be served and monitored.

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
1. sets up a clean Linux environment with Python 3.9
2. installs the dependencies from `requirements.txt`
3. lints the code with flake8
4. runs the test suite with pytest

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
├── pytest.ini                 # pytest configuration
├── requirements.txt           # pinned dependencies
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

## Next steps

- Serve the champion model through a REST API with FastAPI
- Package the service in a Docker container
- Run drift checks automatically on new incoming data and trigger retraining when key features drift
- Compare additional models (e.g. logistic regression, gradient boosting)

## License

This project is released under the MIT License.