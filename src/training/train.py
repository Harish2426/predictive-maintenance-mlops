import os

import pandas as pd
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)


DATA_PATH = "data/raw/ai4i2020.csv"
MODEL_DIR = "artifacts"


def load_data():
    df = pd.read_csv(DATA_PATH)

    columns_to_drop = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def main():

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = load_data()

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    # -----------------------------
    # 2. Define features
    # -----------------------------
    categorical_features = ["Type"]

    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # -----------------------------
    # 3. Preprocessing
    # -----------------------------
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # -----------------------------
    # 4. Model
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -----------------------------
    # 5. Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    # -----------------------------
    # 6. MLflow experiment
    # -----------------------------
    mlflow.set_experiment("predictive-maintenance")

    with mlflow.start_run():

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_probability = pipeline.predict_proba(X_test)[:, 1]

        # -----------------------------
        # 7. Evaluation
        # -----------------------------
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )

        pr_auc = average_precision_score(
            y_test,
            y_probability,
        )

        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        f1 = report["1"]["f1-score"]

        # -----------------------------
        # 8. Log parameters
        # -----------------------------
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        # -----------------------------
        # 9. Log metrics
        # -----------------------------
        mlflow.log_metric("precision_failure", precision)
        mlflow.log_metric("recall_failure", recall)
        mlflow.log_metric("f1_failure", f1)
        mlflow.log_metric("pr_auc", pr_auc)

        # -----------------------------
        # 10. Log model to MLflow
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        # -----------------------------
        # 11. Save model locally
        # -----------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = os.path.join(
            MODEL_DIR,
            "model.pkl",
        )

        import joblib

        joblib.dump(
            pipeline,
            model_path,
        )

        # -----------------------------
        # 12. Print results
        # -----------------------------
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
            )
        )

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nPR-AUC:", round(pr_auc, 4))

        print("\nMLflow Run ID:")
        print(mlflow.active_run().info.run_id)

        print("\nModel saved to:")
        print(model_path)


if __name__ == "__main__":
    main()