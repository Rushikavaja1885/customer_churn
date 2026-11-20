"""Preprocessing utilities for the Telco Customer Churn project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
DEFAULT_DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train and test partitions."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_telco_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the IBM Telco Customer Churn CSV from disk.

    Args:
        data_path: Path to the raw Kaggle CSV file.

    Returns:
        Raw Telco churn dataframe.

    Raises:
        FileNotFoundError: If the CSV file is missing.
        ValueError: If the file is empty.
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    return frame


def clean_telco_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean Telco data and normalize target values.

    The Kaggle dataset stores `TotalCharges` as text and uses a blank string for
    a small number of new customers. This function converts the column to
    numeric and keeps missing values for downstream imputers.

    Args:
        frame: Raw Telco dataframe.

    Returns:
        Cleaned dataframe with a binary churn target.

    Raises:
        ValueError: If the target column is missing or contains unknown values.
    """

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    cleaned = frame.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned = cleaned.drop_duplicates()

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    target_values = set(cleaned[TARGET_COLUMN].dropna().unique())
    if target_values <= {0, 1}:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    elif target_values <= {"No", "Yes"}:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    else:
        raise ValueError(
            f"Unexpected values in '{TARGET_COLUMN}': {sorted(target_values)}. "
            "Expected Yes/No or 0/1."
        )

    return cleaned


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features and target.

    Args:
        frame: Cleaned Telco dataframe containing `Churn`.

    Returns:
        Tuple of feature dataframe and target series.
    """

    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in frame.columns:
        drop_columns.append(ID_COLUMN)

    X = frame.drop(columns=drop_columns)
    y = frame[TARGET_COLUMN].astype(int)
    return X, y


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature names.

    Args:
        frame: Feature dataframe.

    Returns:
        Numeric column names and categorical column names.
    """

    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in frame.columns if column not in set(numeric_columns)
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical columns.

    Args:
        frame: Training feature dataframe used to infer column types.

    Returns:
        ColumnTransformer with imputation, one-hot encoding, and scaling.
    """

    numeric_columns, categorical_columns = infer_feature_types(frame)

    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_dataset_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """Create a stratified train/test split from a cleaned dataframe.

    Args:
        frame: Cleaned Telco dataframe.
        test_size: Fraction of rows held out for testing.
        random_state: Random seed for reproducible splitting.

    Returns:
        DatasetSplit containing feature and target partitions.
    """

    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_model_pipeline(
    model: object,
    training_frame: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
    use_smote: bool = True,
) -> Pipeline:
    """Build an imbalanced-learn pipeline with preprocessing and a classifier.

    Args:
        model: Scikit-learn compatible classifier.
        training_frame: Feature dataframe used to infer preprocessing columns.
        random_state: Random seed passed to SMOTE.
        use_smote: Whether to add SMOTE after preprocessing.

    Returns:
        Imbalanced-learn pipeline ready for fitting.
    """

    steps: list[tuple[str, object]] = [("preprocessor", build_preprocessor(training_frame))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("classifier", model))
    return Pipeline(steps=steps)


def save_pipeline(pipeline: Pipeline, output_path: str | Path) -> Path:
    """Serialize a fitted pipeline to disk.

    Args:
        pipeline: Fitted imbalanced-learn pipeline.
        output_path: Destination path for the serialized pipeline.

    Returns:
        Path where the pipeline was saved.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(model_path: str | Path) -> Pipeline:
    """Load a serialized pipeline from disk.

    Args:
        model_path: Path to a `.pkl` or `.joblib` model artifact.

    Returns:
        Loaded imbalanced-learn pipeline.

    Raises:
        FileNotFoundError: If the model artifact is missing.
    """

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}.")
    return joblib.load(path)


def save_split(split: DatasetSplit, output_dir: str | Path) -> dict[str, Path]:
    """Save train and test partitions as CSV files.

    Args:
        split: DatasetSplit to persist.
        output_dir: Destination directory.

    Returns:
        Mapping of artifact names to saved paths.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "X_train": directory / "X_train.csv",
        "X_test": directory / "X_test.csv",
        "y_train": directory / "y_train.csv",
        "y_test": directory / "y_test.csv",
    }
    split.X_train.to_csv(artifacts["X_train"], index=False)
    split.X_test.to_csv(artifacts["X_test"], index=False)
    split.y_train.to_csv(artifacts["y_train"], index=False)
    split.y_test.to_csv(artifacts["y_test"], index=False)
    return artifacts


def validate_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that a dataframe contains required columns.

    Args:
        frame: Dataframe to validate.
        required_columns: Column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preprocessing."""

    parser = argparse.ArgumentParser(description="Prepare Telco churn train/test splits.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    """Run preprocessing from the command line."""

    args = parse_args()
    raw = load_telco_data(args.data_path)
    cleaned = clean_telco_data(raw)
    split = create_dataset_split(
        cleaned,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    artifacts = save_split(split, args.output_dir)

    print("Saved preprocessing artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
