import os
import json
import logging
import joblib
import pandas as pd
import tarfile

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

MODEL_NAME = "customer_churn_pipeline.pkl"

TARGET_COLUMN = "Churn"

EVALUATION_FILE = "evaluation.json"

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# SageMaker Directories
# ==========================================================

TEST_DIR = "/opt/ml/processing/test"
MODEL_DIR = "/opt/ml/model"
EVALUATION_DIR = "/opt/ml/processing/evaluation"

TARGET_COLUMN = "Churn"

# ==========================================================
# Load Test Dataset
# ==========================================================

def load_test_data():

    test_path = os.path.join(TEST_DIR, "test.csv")

    logger.info(f"Loading test dataset from: {test_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
    )

    df = pd.read_csv(test_path)

    logger.info(f"Test Dataset Shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
    )

    X_test = df.drop(columns=[TARGET_COLUMN])

    y_test = df[TARGET_COLUMN]

    return X_test, y_test


# ==========================================================
# Load Trained Pipeline
# ==========================================================

def load_pipeline():

    model_path = os.path.join(
        MODEL_DIR,
        MODEL_NAME
    )

    # --------------------------------------------------
    # Local execution
    # --------------------------------------------------
    if os.path.exists(model_path):

        logger.info(f"Loading pipeline from: {model_path}")

        return joblib.load(model_path)

    # --------------------------------------------------
    # SageMaker execution
    # --------------------------------------------------
    model_archive = os.path.join(
        MODEL_DIR,
        "model.tar.gz"
    )

    if os.path.exists(model_archive):

        logger.info("Extracting model.tar.gz...")

        with tarfile.open(model_archive) as tar:
            tar.extractall(path=MODEL_DIR)

        logger.info(f"Loading pipeline from: {model_path}")

        return joblib.load(model_path)

    raise FileNotFoundError(
        "Neither customer_churn_pipeline.pkl nor model.tar.gz was found."
    )

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(pipeline, X_test, y_test):

    logger.info("Running predictions...")

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    logger.info("=" * 60)
    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"ROC AUC  : {roc_auc:.4f}")
    logger.info("=" * 60)

    evaluation = {
        "classification_metrics": {
            "accuracy": {
                "value": float(accuracy)
            },
            "precision": {
                "value": float(precision)
            },
            "recall": {
                "value": float(recall)
            },
            "f1_score": {
                "value": float(f1)
            },
            "roc_auc": {
                "value": float(roc_auc)
            }
        },
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        }
    }

    return evaluation


# ==========================================================
# Save Evaluation Report
# ==========================================================

def save_evaluation(evaluation):

    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    '''
    evaluation_path = os.path.join(
        EVALUATION_DIR,
        "evaluation.json"
    )'''

    evaluation_path = os.path.join(
        EVALUATION_DIR,
        EVALUATION_FILE
)

    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=4)

    logger.info(f"Evaluation report saved to: {evaluation_path}")


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Customer Churn Model Evaluation Started")
    logger.info("=" * 60)

    X_test, y_test = load_test_data()

    pipeline = load_pipeline()

    evaluation = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    save_evaluation(evaluation)

    logger.info("=" * 60)
    logger.info("Evaluation Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()