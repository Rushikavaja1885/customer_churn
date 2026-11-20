"""Prediction helpers for Telco customer churn models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from preprocess import ID_COLUMN, TARGET_COLUMN, load_pipeline


DEFAULT_MODEL_PATH = Path("models/xgboost_churn_model.pkl")


def prepare_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns before prediction.

    Args:
        frame: Raw customer dataframe.

    Returns:
        Feature dataframe suitable for the saved model pipeline.
    """

    drop_columns = [column for column in [ID_COLUMN, TARGET_COLUMN] if column in frame.columns]
    return frame.drop(columns=drop_columns)


def predict_churn(
    pipeline: Any,
    frame: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict churn probabilities and classes for customers.

    Args:
        pipeline: Fitted model pipeline.
        frame: Customer dataframe with Telco feature columns.
        threshold: Probability threshold for converting scores to labels.

    Returns:
        Dataframe containing customer identifiers, churn probability, and predicted label.
    """

    features = prepare_prediction_frame(frame)
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": predictions,
            "predicted_label": ["Yes" if value == 1 else "No" for value in predictions],
        }
    )
    if ID_COLUMN in frame.columns:
        output.insert(0, ID_COLUMN, frame[ID_COLUMN].values)
    return output


def predict_from_csv(
    model_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict churn for customers in a CSV file.

    Args:
        model_path: Path to a fitted pipeline artifact.
        input_csv: Input customer CSV path.
        output_csv: Optional destination for prediction results.
        threshold: Probability threshold for predicted labels.

    Returns:
        Prediction dataframe.
    """

    pipeline = load_pipeline(model_path)
    frame = pd.read_csv(input_csv)
    predictions = predict_churn(pipeline, frame, threshold=threshold)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)

    return predictions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch prediction."""

    parser = argparse.ArgumentParser(description="Predict churn for a customer CSV.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run batch prediction from the command line."""

    args = parse_args()
    predictions = predict_from_csv(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        threshold=args.threshold,
    )
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
