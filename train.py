import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

DATA_PATH = "dataset_phishing.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["url", "status"])
    y = df["status"].map({"legitimate": 0, "phishing": 1})
    return X, y

def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
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
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")