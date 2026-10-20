import mlflow.pyfunc

from train import load_data

MODEL_URI = "models:/phishing-detector@champion"


if __name__ == "__main__":
    model = mlflow.pyfunc.load_model(MODEL_URI)

    X, y = load_data()
    sample = X.sample(5, random_state=1).astype("float64")
    preds = model.predict(sample)

    labels = {0: "legitimate", 1: "phishing"}
    for pred, actual in zip(preds, y.loc[sample.index]):
        print(f"predicted: {labels[pred]:<10}  actual: {labels[actual]}")