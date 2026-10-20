import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = "dataset_phishing.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["url", "status"])
    y = df["status"].map({"legitimate": 0, "phishing": 1})
    return X, y

def build_model(n_estimators=100, max_depth=None, min_samples_split=2):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )),
    ])

def train_model(X_train, y_train, **params):
    model = build_model(**params)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return{
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

if __name__ == "__main__":
    n_estimators = 100

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("phishing-detection")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", 0.2)

        model = train_model(X_train, y_train, n_estimators=n_estimators)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head().astype("float64"))

        for name, value in metrics.items():
            print(f"{name}: {value:.3f}")
