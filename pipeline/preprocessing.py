import os
import logging

import pandas as pd

from sklearn.model_selection import train_test_split

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

INPUT_DIR = "/opt/ml/processing/input"
TRAIN_DIR = "/opt/ml/processing/train"
TEST_DIR = "/opt/ml/processing/test"

DATASET_NAME = "customer_churn_dataset.csv"

TARGET_COLUMN = "Churn"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    dataset_path = os.path.join(INPUT_DIR, DATASET_NAME)

    logger.info(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)

    logger.info(f"Dataset loaded successfully.")

    logger.info(f"Dataset Shape: {df.shape}")

    return df


# ==========================================================
# Clean Dataset
# ==========================================================

def clean_dataset(df):

    logger.info("Cleaning dataset...")

    # Remove customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric and Fill missing values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
    )

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
    )

    # Remove duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        logger.info(f"Removing {duplicates} duplicate rows.")
        df = df.drop_duplicates()

    logger.info("Dataset cleaning completed.")
    logger.info(f"Final dataset shape: {df.shape}")

    return df


# ==========================================================
# Train/Test Split
# ==========================================================

def split_dataset(df):

    logger.info("Splitting dataset...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
    )

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )

    train_df = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    logger.info(f"Training samples : {len(train_df)}")

    logger.info(f"Testing samples  : {len(test_df)}")

    return train_df, test_df


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(train_df, test_df):

    os.makedirs(TRAIN_DIR, exist_ok=True)

    os.makedirs(TEST_DIR, exist_ok=True)

    train_path = os.path.join(TRAIN_DIR, "train.csv")

    test_path = os.path.join(TEST_DIR, "test.csv")

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    logger.info(f"Training dataset saved to {train_path}")

    logger.info(f"Testing dataset saved to {test_path}")


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Starting SageMaker Preprocessing Job")
    logger.info("=" * 60)

    df = load_dataset()

    df = clean_dataset(df)

    train_df, test_df = split_dataset(df)

    save_dataset(train_df, test_df)

    logger.info("=" * 60)
    logger.info("Preprocessing Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()