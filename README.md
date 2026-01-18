# Fraud Detection using Isolation Forest

## Project Overview
This project implements an **unsupervised fraud detection pipeline** using the **Isolation Forest** algorithm to identify anomalous credit card transactions. The main objective is to detect potentially fraudulent transactions without relying on labeled data during training, simulating a realistic production scenario where fraud labels are scarce or delayed.

The project was developed end-to-end, covering data preparation, feature scaling, anomaly scoring, evaluation against known fraud labels, and ranking of the most suspicious transactions.

---

## Dataset
The project uses the **Credit Card Fraud Detection** dataset, which contains anonymized credit card transactions made by European cardholders.

- Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Total transactions: ~284,000
- Fraud cases: ~0.17% of the dataset
- Features: PCA-transformed numerical variables + transaction `Amount`

> **Note**: The dataset file (`creditcard.csv`) is intentionally **not versioned** in this repository due to size constraints and good data engineering practices. To run the project, download the dataset and place it at:

```text
data/creditcard.csv
```

---

## Methodology

1. **Data Loading & Preparation**
   - Removal of non-informative columns (`Time`, `Class`)
   - Separation of features and target (used only for evaluation)

2. **Feature Scaling**
   - Standardization of the `Amount` feature using `StandardScaler`

3. **Modeling (Unsupervised)**
   - Isolation Forest with:
     - `n_estimators = 100`
     - `contamination = auto`
     - Fixed `random_state` for reproducibility

4. **Anomaly Scoring**
   - Continuous anomaly score via `decision_function`
   - Binary anomaly flag (`0 = normal`, `1 = anomaly`)

5. **Evaluation**
   - Comparison against true fraud labels
   - Confusion matrix and classification report
   - Ranking transactions by anomaly score

6. **Top-N Analysis**
   - Evaluation of how many true fraud cases appear among the **Top 1000 most anomalous transactions**

---

## Project Structure

```text
fraud_detection/
│
├── data/                    # Dataset folder (ignored by git)
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection_isolation_forest.ipynb
│
├── src/
│   └── isolation_forest_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## Results & Evaluation

- The Isolation Forest successfully assigns anomaly scores that prioritize fraudulent transactions.
- Although the model is unsupervised, a significant number of true fraud cases appear among the lowest anomaly scores.
- The **Top-N anomaly analysis** demonstrates the model's usefulness as a *prioritization tool* rather than a strict classifier.

Key outputs:
- Confusion Matrix
- Classification Report
- Number of fraud cases in the Top 1000 anomalies

---

## Limitations

- Isolation Forest does not leverage label information during training.
- Performance is sensitive to feature engineering and contamination assumptions.
- Evaluation metrics should be interpreted with caution due to extreme class imbalance.

---

## How to Run

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset and place it in `data/creditcard.csv`
5. Run the pipeline:

```bash
python src/isolation_forest_pipeline.py
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## Author

Developed as a portfolio project focused on **fraud detection**, **unsupervised machine learning**, and **reproducible data pipelines**.