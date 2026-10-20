import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from train import load_data, build_model, train_model, evaluate

N_TRIALS = 20

def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    model = build_model(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1").mean()

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_f1", score)
    
    return score


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("phishing-detection")

    with mlflow.start_run(run_name="optuna-tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS)

        print("Best params:", study.best_params)
        print(f"Best CV F1: {study.best_value:.3f}")

        model = train_model(X_train, y_train, **study.best_params)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_f1", study.best_value)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head().astype("float64"))

        for name, value in metrics.items():
            print(f"test {name}: {value:.3f}")