import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "predictive-maintenance-docker"

MIN_PR_AUC = 0.80
MIN_RECALL = 0.75


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Connect to MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    # --------------------------------------------------------
    # Find experiment
    # --------------------------------------------------------

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found."
        )

    # --------------------------------------------------------
    # Get latest run
    # --------------------------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            "No MLflow runs found."
        )

    run = runs[0]

    run_id = run.info.run_id
    status = run.info.status

    if status != "FINISHED":
        raise RuntimeError(
            f"Latest run is not finished. Status: {status}"
        )

    # --------------------------------------------------------
    # Read metrics
    # --------------------------------------------------------

    pr_auc = run.data.metrics.get("pr_auc")
    recall = run.data.metrics.get("recall_failure")
    model = run.data.params.get("model", "Unknown")

    if pr_auc is None:
        raise RuntimeError(
            "Metric 'pr_auc' not found."
        )

    if recall is None:
        raise RuntimeError(
            "Metric 'recall_failure' not found."
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("=" * 60)
    print("MODEL QUALITY GATE")
    print("=" * 60)

    print("Experiment:", EXPERIMENT_NAME)
    print("Run ID:", run_id)
    print("Model:", model)

    print("PR-AUC:", round(pr_auc, 4))
    print("Recall:", round(recall, 4))

    print("\nRequirements:")
    print("PR-AUC >=", MIN_PR_AUC)
    print("Recall  >=", MIN_RECALL)

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    pr_auc_passed = pr_auc >= MIN_PR_AUC
    recall_passed = recall >= MIN_RECALL

    print()

    if pr_auc_passed:
        print("✅ PR-AUC requirement PASSED")
    else:
        print("❌ PR-AUC requirement FAILED")

    if recall_passed:
        print("✅ Recall requirement PASSED")
    else:
        print("❌ Recall requirement FAILED")

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if pr_auc_passed and recall_passed:
        print("✅ QUALITY GATE PASSED")
        print("Model is eligible for production.")
        print("=" * 60)
        return

    print("❌ QUALITY GATE FAILED")
    print("Model is NOT eligible for production.")
    print("=" * 60)

    raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()