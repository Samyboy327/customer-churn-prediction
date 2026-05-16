import os
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix
)


def train_model(X_train, y_train):

    print("\nTraining XGBoost Model...\n")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Model...\n")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, predictions)

    print("=" * 50)
    print("XGBoost Model Performance")
    print("=" * 50)

    print(f"Accuracy Score : {accuracy:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return accuracy, f1, roc_auc


def save_model(model):

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/xgboost_model.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path}")


def log_with_mlflow(model, accuracy, f1, roc_auc):

    # Connect to MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Create experiment
    mlflow.set_experiment("Customer_Churn_Experiment")

    with mlflow.start_run() as run:

        # Parameters
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name ="model"
        )

        print("\nMLflow logging completed.")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="CustomerChurnModel"
        )

        print("\nModel registered successfully.")
        print(f"Model Version: {registered_model.version}")



