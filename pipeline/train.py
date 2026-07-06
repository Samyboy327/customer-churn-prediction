import os
import logging
import joblib
import pandas as pd
#from config import MODEL_NAME

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

MODEL_NAME = "customer_churn_pipeline.pkl"

TARGET_COLUMN = "Churn"

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

TRAIN_DIR = "/opt/ml/input/data/train"
MODEL_DIR = "/opt/ml/model"

TARGET_COLUMN = "Churn"

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    train_path = os.path.join(TRAIN_DIR, "train.csv")

    logger.info(f"Loading training dataset from: {train_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
    )

    df = pd.read_csv(train_path)

    logger.info(f"Dataset Shape: {df.shape}")

    return df

# ==========================================================
# Prepare Features and Target
# ==========================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
    )

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    return X, y


# ==========================================================
# Build Training Pipeline
# ==========================================================

def build_pipeline(X):

    logger.info("Building preprocessing pipeline...")

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    logger.info(f"Categorical Features: {categorical_features}")

    logger.info(f"Numerical Features: {numerical_features}")

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "categorical",

                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False

                ),

                categorical_features

            ),

            (

                "numerical",

                "passthrough",

                numerical_features

            )

        ]

    )

    classifier = XGBClassifier(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1

    )

    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),

            (

                "classifier",

                classifier

            )

        ]

    )

    logger.info("Pipeline created successfully.")

    return pipeline


# ==========================================================
# Train Model
# ==========================================================

def train_model(pipeline, X, y):

    logger.info("Training model...")

    pipeline.fit(X, y)

    #logger.info(f"Training samples: {len(X)}")

    logger.info("Training completed.")

    return pipeline


# ==========================================================
# Evaluate Training Performance
# ==========================================================

def evaluate_training(pipeline, X, y):

    logger.info("Evaluating training performance...")

    predictions = pipeline.predict(X)

    probabilities = pipeline.predict_proba(X)[:, 1]

    metrics = {

        "Accuracy": accuracy_score(y, predictions),

        "Precision": precision_score(y, predictions),

        "Recall": recall_score(y, predictions),

        "F1 Score": f1_score(y, predictions),

        "ROC AUC": roc_auc_score(y, probabilities)

    }

    logger.info("=" * 50)

    for metric, value in metrics.items():

        logger.info(f"{metric}: {value:.4f}")

    logger.info("=" * 50)

    return metrics


# ==========================================================
# Save Pipeline
# ==========================================================

def save_pipeline(pipeline):

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(

        MODEL_DIR,
        MODEL_NAME

    )

    joblib.dump(

        pipeline,

        model_path

    )

    logger.info(f"Pipeline saved to: {model_path}")

    #logger.info(f"Model size: {os.path.getsize(model_path)/1024:.2f} KB")



# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)

    logger.info("Customer Churn Training Started")

    logger.info("=" * 60)

    df = load_dataset()

    X, y = prepare_data(df)

    pipeline = build_pipeline(X)

    pipeline = train_model(

        pipeline,

        X,

        y

    )

    evaluate_training(

        pipeline,

        X,

        y

    )

    save_pipeline(

        pipeline

    )

    logger.info("=" * 60)

    logger.info("Training Completed Successfully")

    logger.info("=" * 60)


if __name__ == "__main__":

    main()