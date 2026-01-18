import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report


def run_isolation_forest(
    data_path: str,
    filename: str = "creditcard.csv",
    n_estimators: int = 100,
    contamination: str = "auto",
    random_state: int = 13,
    top_n: int = 1000,
):
    """
    Run an Isolation Forest pipeline for fraud detection.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the dataset.
    filename : str
        CSV file name.
    n_estimators : int
        Number of trees in Isolation Forest.
    contamination : str or float
        Proportion of outliers.
    random_state : int
        Random seed.
    top_n : int
        Number of lowest scores used for top-N anomaly analysis.

    Returns
    -------
    dict
        Dictionary with confusion matrix, classification report
        and fraud count in top-N anomalies.
    """

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(os.path.join(data_path, filename))

    X = df.drop(["Time", "Class"], axis=1)
    y = df["Class"]

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_feat = X.copy()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=random_state,
    )

    model.fit(X_feat)

    # ------------------------------------------------------------------
    # Scoring & prediction
    # ------------------------------------------------------------------
    X["score"] = model.decision_function(X_feat)
    X["anomaly"] = model.predict(X_feat)
    X["anomaly"] = X["anomaly"].map({1: 0, -1: 1})

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    X_eval = X.copy()
    X_eval["Class"] = y

    cm = confusion_matrix(X_eval["Class"], X_eval["anomaly"])
    report = classification_report(
        X_eval["Class"],
        X_eval["anomaly"],
        target_names=["Normal", "Fraud"],
    )

    # ------------------------------------------------------------------
    # Top-N anomaly analysis
    # ------------------------------------------------------------------
    X_sorted = X_eval.sort_values(by="score", ascending=True)
    fraud_in_top_n = X_sorted.head(top_n)["Class"].value_counts().get(1, 0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    results = {
        "confusion_matrix": cm,
        "classification_report": report,
        "fraud_in_top_n": fraud_in_top_n,
    }

    return results


if __name__ == "__main__":
    DATA_PATH = r"D:\project\fraud_detection\data"

    results = run_isolation_forest(data_path=DATA_PATH)

    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print(
        f"\nNumber of fraud cases in top 1000 anomalies: "
        f"{results['fraud_in_top_n']}"
    )
