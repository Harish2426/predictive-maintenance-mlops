import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "predictive-maintenance-docker"

MODEL_NAME = "predictive-maintenance-model"

MIN_PR_AUC = 0.80
MIN_RECALL = 0.75


def main():

    client = MlflowClient()

    # -----------------------------
    # 1. Find experiment
    # -----------------------------
    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found."
        )

    # -----------------------------
    # 2. Get latest successful run
    # -----------------------------
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            "No successful MLflow runs found."
        )

    run = runs[0]

    run_id = run.info.run_id

    # -----------------------------
    # 3. Read metrics
    # -----------------------------
    pr_auc = run.data.metrics.get(
        "pr_auc",
        0
    )

    recall = run.data.metrics.get(
        "recall_failure",
        0
    )

    model_type = run.data.params.get(
        "model",
        "Unknown"
    )

    print("=" * 50)
    print("MODEL REGISTRATION")
    print("=" * 50)

    print(f"Run ID: {run_id}")
    print(f"Model: {model_type}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall: {recall:.4f}")

    # -----------------------------
    # 4. Quality gate
    # -----------------------------
    if pr_auc < MIN_PR_AUC:
        raise RuntimeError(
            "❌ Model rejected: PR-AUC below threshold."
        )

    if recall < MIN_RECALL:
        raise RuntimeError(
            "❌ Model rejected: Recall below threshold."
        )

    print("\n✅ Quality gate passed.")

    # -----------------------------
    # 5. Model URI
    # -----------------------------
    model_uri = f"runs:/{run_id}/model"

    # -----------------------------
    # 6. Register model
    # -----------------------------
    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n✅ Model registered successfully.")

    print(f"Model name: {result.name}")
    print(f"Model version: {result.version}")


if __name__ == "__main__":
    main()