from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from train import load_data


def make_drift_report(reference, current, path):
    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    result.save_html(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scenario 1: new data similar to the training data
    make_drift_report(X_train, X_test, "drift_report_no_drift.html")

    # Scenario 2: simulated drift (phishers start using longer, more complex URLs)
    X_drifted = X_test.copy()
    X_drifted["length_url"] = X_drifted["length_url"] * 2
    X_drifted["nb_dots"] = X_drifted["nb_dots"] + 3
    make_drift_report(X_train, X_drifted, "drift_report_drift.html")