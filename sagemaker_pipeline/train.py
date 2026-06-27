import os
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    # SageMaker training directory
    train_path = "/opt/ml/input/data/train"

    # Load dataset
    df = pd.read_csv(
        os.path.join(
            train_path,
            "customer_churn_dataset.csv"
        )
    )

    # Drop customerID
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encode categorical columns
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")

    # Save model
    model_dir = "/opt/ml/model"

    joblib.dump(
        model,
        os.path.join(model_dir, "model.pkl")
    )

    print("Model saved successfully.")