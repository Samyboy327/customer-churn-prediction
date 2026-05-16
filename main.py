import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from src.ingest import load_data
from src.preprocess import preprocess_data


from train import (
    train_model,
    evaluate_model,
    save_model,
    log_with_mlflow
)

# Load dataset
df = load_data("data/customer_churn_dataset.csv")

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy, f1, roc_auc = evaluate_model(
    model,
    X_test,
    y_test
)

# Save model
save_model(model)

# Log to MLflow
log_with_mlflow(
    model,
    accuracy,
    f1,
    roc_auc
)

print("\nPipeline completed successfully.")