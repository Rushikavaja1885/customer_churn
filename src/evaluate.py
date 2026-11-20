"""Evaluation and SHAP explanation utilities for churn models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from preprocess import (
    DEFAULT_DATA_PATH,
    clean_telco_data,
    create_dataset_split,
    load_pipeline,
    load_telco_data,
)


DEFAULT_MODEL_PATH = Path("models/xgboost_churn_model.pkl")


def evaluate_classifier(pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted classifier pipeline on a holdout set.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout features.
        y_test: Holdout labels.

    Returns:
        Metrics dictionary containing ROC-AUC, F1, confusion matrix, and report.
    """

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            output_dict=True,
        ),
    }


def save_confusion_matrix(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str | Path,
) -> Path:
    """Save a confusion matrix plot.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout features.
        y_test: Holdout labels.
        output_path: Destination image path.

    Returns:
        Saved image path.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = pipeline.predict(X_test)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.figure(figsize=(12, 8))
    plt.savefig(output, dpi=100)
    plt.close()
    return output


def save_roc_curve(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str | Path,
) -> Path:
    """Save a ROC curve plot.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout features.
        y_test: Holdout labels.
        output_path: Destination image path.

    Returns:
        Saved image path.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.figure(figsize=(12, 8))
    plt.savefig(output, dpi=100)
    plt.close()
    return output


def transformed_features(pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Transform raw features with the fitted preprocessor.

    Args:
        pipeline: Fitted pipeline containing a `preprocessor` step.
        frame: Raw feature dataframe.

    Returns:
        Transformed feature matrix and feature names.
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(frame)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return transformed, feature_names


def shap_values_for_sample(
    pipeline: Any,
    frame: pd.DataFrame,
    max_rows: int = 250,
) -> shap.Explanation:
    """Calculate SHAP values for a sample of rows.

    Args:
        pipeline: Fitted classifier pipeline.
        frame: Raw feature dataframe.
        max_rows: Maximum rows used for SHAP plotting.

    Returns:
        SHAP Explanation object.
    """

    sample = frame.sample(n=min(max_rows, len(frame)), random_state=42)
    matrix, feature_names = transformed_features(pipeline, sample)
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.Explainer(classifier, matrix, feature_names=feature_names)
    return explainer(matrix)


def save_shap_summary(
    pipeline: Any,
    X_test: pd.DataFrame,
    output_path: str | Path,
    max_rows: int = 250,
) -> Path:
    """Save a SHAP summary beeswarm plot.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout features.
        output_path: Destination image path.
        max_rows: Maximum rows used for SHAP plotting.

    Returns:
        Saved image path.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    explanations = shap_values_for_sample(pipeline, X_test, max_rows=max_rows)
    shap.summary_plot(explanations, show=False, max_display=20)
    plt.tight_layout()
    plt.figure(figsize=(12, 8))
    plt.savefig(output, dpi=100)
    plt.close()
    return output


def save_shap_waterfall(
    pipeline: Any,
    X_test: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a SHAP waterfall plot for the highest-risk test customer.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout features.
        output_path: Destination image path.

    Returns:
        Saved image path.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    highest_risk_position = int(np.argmax(probabilities))
    customer = X_test.iloc[[highest_risk_position]]
    matrix, feature_names = transformed_features(pipeline, customer)
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.Explainer(classifier, matrix, feature_names=feature_names)
    explanation = explainer(matrix)[0]

    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.figure(figsize=(12, 8))
    plt.savefig(output, dpi=100)  
    plt.close()
    return output


def evaluate_saved_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = Path("outputs/evaluation"),
) -> dict[str, Any]:
    """Evaluate a saved model and write metrics plus plots.

    Args:
        model_path: Path to a fitted pipeline artifact.
        data_path: Path to the raw Telco CSV.
        output_dir: Directory for metrics and plots.

    Returns:
        Metrics dictionary with plot artifact paths.
    """

    pipeline = load_pipeline(model_path)
    raw = load_telco_data(data_path)
    cleaned = clean_telco_data(raw)
    split = create_dataset_split(cleaned)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_classifier(pipeline, split.X_test, split.y_test)
    metrics["artifacts"] = {
        "confusion_matrix": str(
            save_confusion_matrix(
                pipeline,
                split.X_test,
                split.y_test,
                output / "confusion_matrix.png",
            )
        ),
        "roc_curve": str(save_roc_curve(pipeline, split.X_test, split.y_test, output / "roc_curve.png")),
        "shap_summary": str(save_shap_summary(pipeline, split.X_test, output / "shap_summary.png")),
        "shap_waterfall": str(
            save_shap_waterfall(pipeline, split.X_test, output / "shap_waterfall.png")
        ),
    }

    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a saved churn model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    return parser.parse_args()


def main() -> None:
    """Run evaluation from the command line."""

    args = parse_args()
    metrics = evaluate_saved_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
