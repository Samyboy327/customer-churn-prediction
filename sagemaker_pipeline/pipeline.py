import sagemaker

from sagemaker.xgboost.estimator import XGBoost


# SageMaker session
session = sagemaker.Session()

# IAM role
role = sagemaker.get_execution_role()

# S3 bucket
bucket = "customer-churn-mlops-bucket"

# XGBoost estimator
estimator = XGBoost(
    entry_point="src/sagemaker_pipeline/train.py",

    framework_version="1.7-1",

    instance_type="ml.m5.large",

    instance_count=1,

    role=role,

    output_path=f"s3://customer-churn-mlops-bucket/models/",

    sagemaker_session=session
)

# Launch training job
estimator.fit(
    {
        "train":
        f"s3://customer-churn-mlops-bucket/data/"
    }
)

print("Training job completed.")