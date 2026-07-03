"""
Configuration for the Customer Churn SageMaker Pipeline
"""

import boto3
import sagemaker


# ==========================================================
# Framework Versions
# ==========================================================

SKLEARN_FRAMEWORK_VERSION = "1.2-1"

PYTHON_VERSION = "py3"

SOURCE_DIR = "pipeline"

# ==========================================================
# Scripts
# ==========================================================

PREPROCESSING_SCRIPT = "preprocessing.py"

TRAIN_SCRIPT = "train.py"

EVALUATE_SCRIPT = "evaluate.py"

# ==========================================================
# Model
# ==========================================================

MODEL_NAME = "customer_churn_pipeline.pkl"

EVALUATION_FILE = "evaluation.json"

# ==========================================================
# AWS Session
# ==========================================================

boto_session = boto3.Session()

REGION = boto_session.region_name

sagemaker_session = sagemaker.Session(
    boto_session=boto_session
)

# ==========================================================
# IAM Role
# ==========================================================

ROLE = "arn:aws:iam::<459858400546>:role/<SageMakerExecutionRole>"

# ==========================================================
# S3 Bucket
# ==========================================================

BUCKET_NAME = "churn-prediction-mlops-bucket"

# ==========================================================
# Pipeline
# ==========================================================

PIPELINE_NAME = "CustomerChurnPipeline"

MODEL_PACKAGE_GROUP_NAME = "CustomerChurnModelGroup"

# ==========================================================
# Dataset
# ==========================================================

DATASET_NAME = "customer_churn_dataset.csv"

RAW_DATA_URI = (
    f"s3://{BUCKET_NAME}/data/{DATASET_NAME}"
)

# ==========================================================
# Pipeline Outputs
# ==========================================================

PROCESSING_OUTPUT_URI = (
    f"s3://{BUCKET_NAME}/processed"
)

MODEL_OUTPUT_URI = (
    f"s3://{BUCKET_NAME}/models"
)

EVALUATION_OUTPUT_URI = (
    f"s3://{BUCKET_NAME}/evaluation"
)
# ==========================================================
# Compute
# ==========================================================

PROCESSING_INSTANCE_TYPE = "ml.m5.large"

TRAINING_INSTANCE_TYPE = "ml.m5.large"

EVALUATION_INSTANCE_TYPE = "ml.m5.large"

INSTANCE_COUNT = 1

# ==========================================================
# Model Approval
# ==========================================================

ACCURACY_THRESHOLD = 0.80

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_STATE = 42