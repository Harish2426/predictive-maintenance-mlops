import os

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/ai4i2020.csv"
MODEL_DIR = "artifacts"

# Connect to the running MLflow server
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# New clean experiment
EXPERIMENT_NAME = "predictive-maintenance-docker"


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ============================================================
# LOAD DATA
# ============================================================

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

    return df.drop(columns=columns_to_drop)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data()

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    # --------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------

    categorical_features = [
        "Type"
    ]

    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # --------------------------------------------------------
    # 3. Preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # --------------------------------------------------------
    # 4. XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=28.5,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # 5. Complete ML pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # --------------------------------------------------------
    # 6. Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    # --------------------------------------------------------
    # 7. Configure MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(EXPERIMENT_NAME)

    # --------------------------------------------------------
    # 8. Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run() as run:

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = pipeline.predict(X_test)

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            "XGBoost",
        )

        mlflow.log_param(
            "n_estimators",
            300,
        )

        mlflow.log_param(
            "max_depth",
            5,
        )

        mlflow.log_param(
            "learning_rate",
            0.05,
        )

        mlflow.log_param(
            "subsample",
            0.8,
        )

        mlflow.log_param(
            "colsample_bytree",
            0.8,
        )

        mlflow.log_param(
            "scale_pos_weight",
            28.5,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "precision_failure",
            precision,
        )

        mlflow.log_metric(
            "recall_failure",
            recall,
        )

        mlflow.log_metric(
            "f1_failure",
            f1,
        )

        mlflow.log_metric(
            "pr_auc",
            pr_auc,
        )

        # ----------------------------------------------------
        # Log model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

        # ----------------------------------------------------
        # Save local model
        # ----------------------------------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True,
        )

        model_path = os.path.join(
            MODEL_DIR,
            "xgboost_model.pkl",
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

        print(
            classification_report(
                y_test,
                y_pred,
            )
        )

        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)

        print(
            confusion_matrix(
                y_test,
                y_pred,
            )
        )

        print("\n" + "=" * 60)
        print("MODEL METRICS")
        print("=" * 60)

        print(
            "PR-AUC:",
            round(pr_auc, 4),
        )

        print(
            "Failure Precision:",
            round(precision, 4),
        )

        print(
            "Failure Recall:",
            round(recall, 4),
        )

        print(
            "Failure F1:",
            round(f1, 4),
        )

        print("\n" + "=" * 60)
        print("MLFLOW")
        print("=" * 60)

        print(
            "Tracking URI:",
            mlflow.get_tracking_uri(),
        )

        print(
            "Experiment:",
            EXPERIMENT_NAME,
        )

        print(
            "Run ID:",
            run.info.run_id,
        )

        print("\nModel saved locally to:")
        print(model_path)

        print("\nTraining completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()