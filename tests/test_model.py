import pytest
from sklearn.model_selection import train_test_split
from train import load_data, train_model, evaluate

@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    return train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)

@pytest.fixture(scope="module")
def model(data):
    X_train, X_test, y_train, y_test = data
    return train_model(X_train, y_train)

def test_data_has_expected_columns():
    X, y = load_data()
    assert X.shape[1] == 87
    assert "url" not in X.columns
    assert "status" not in X.columns
    assert len(X) == len(y)

def test_labels_are_binary():
    _, y = load_data()
    assert y.isnull().sum() == 0
    assert set(y.unique()) == {0, 1}

def test_no_missing_values():
    X, _ = load_data()
    assert X.isnull().sum().sum() == 0

def test_predictions_are_valid(model, data):
    _, X_test, _, _ = data
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_model_performance(model, data):
    _, X_test, _, y_test = data
    metrics = evaluate(model, X_test, y_test)
    assert metrics["f1"] > 0.9
    assert metrics["recall"] > 0.9

def test_model_handles_missing_values(model, data):
    _, X_test, _, _ = data
    X_missing = X_test.head(5).astype("float64")
    X_missing.iloc[0,0] = float("nan")
    preds = model.predict(X_missing)
    assert len(preds) == 5