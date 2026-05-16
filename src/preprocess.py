import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(df):

    # Drop customerID
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors='coerce'
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns

    le_dict = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Save encoders
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(
        le_dict,
        "artifacts/label_encoders.pkl"
    )

    # Features & target
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

    return X_train, X_test, y_train, y_test