"""Model training for Telco customer churn prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from preprocess import (
    DEFAULT_DATA_PATH,
    DEFAULT_RANDOM_STATE,
    build_model_pipeline,
    clean_telco_data,
    create_dataset_split,
    load_telco_data,
    save_pipeline,
)


DEFAULT_MODEL_DIR = Path("models")


def build_logistic_regression(random_state: int = DEFAULT_RANDOM_STATE) -> LogisticRegression:
    """Create the baseline Logistic Regression classifier.

    Args:
        random_state: Random seed for reproducible solver behavior.

    Returns:
        Configured LogisticRegression classifier.
    """

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )


def build_xgboost_classifier(
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> XGBClassifier:
    """Create an XGBoost classifier with project defaults.

    Args:
        params: Optional hyperparameters from Optuna.
        random_state: Random seed for reproducible training.

    Returns:
        Configured XGBClassifier.
    """

    base_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)
    return XGBClassifier(**base_params)


def score_pipeline(pipeline: object, X_test: object, y_test: object) -> dict[str, float]:
    """Calculate ROC-AUC and F1 for a fitted classifier pipeline.

    Args:
        pipeline: Fitted classifier pipeline.
        X_test: Holdout feature data.
        y_test: Holdout labels.

    Returns:
        Dictionary with ROC-AUC and F1 scores.
    """

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions)),
    }


def tune_xgboost(
    X_train: object,
    y_train: object,
    n_trials: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Tune XGBoost hyperparameters with Optuna.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_trials: Number of Optuna trials.
        random_state: Random seed for sampler and model.

    Returns:
        Best hyperparameter dictionary.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 650, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = build_xgboost_classifier(params=params, random_state=random_state)
        pipeline = build_model_pipeline(model, X_train, random_state=random_state, use_smote=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params)


def train_models(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    n_trials: int = 40,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train baseline and tuned models, then save artifacts.

    Args:
        data_path: Path to the raw Telco CSV.
        model_dir: Directory for saved model artifacts.
        n_trials: Number of Optuna trials for XGBoost tuning.
        random_state: Random seed for reproducibility.

    Returns:
        Training metrics and saved artifact paths.
    """

    raw = load_telco_data(data_path)
    cleaned = clean_telco_data(raw)
    split = create_dataset_split(cleaned, random_state=random_state)
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)

    logistic_pipeline = build_model_pipeline(
        build_logistic_regression(random_state=random_state),
        split.X_train,
        random_state=random_state,
        use_smote=True,
    )
    logistic_pipeline.fit(split.X_train, split.y_train)
    logistic_metrics = score_pipeline(logistic_pipeline, split.X_test, split.y_test)
    logistic_path = save_pipeline(
        logistic_pipeline,
        model_directory / "logistic_regression_churn_model.pkl",
    )

    best_params = tune_xgboost(
        split.X_train,
        split.y_train,
        n_trials=n_trials,
        random_state=random_state,
    )
    xgboost_pipeline = build_model_pipeline(
        build_xgboost_classifier(params=best_params, random_state=random_state),
        split.X_train,
        random_state=random_state,
        use_smote=True,
    )
    xgboost_pipeline.fit(split.X_train, split.y_train)
    xgboost_metrics = score_pipeline(xgboost_pipeline, split.X_test, split.y_test)
    xgboost_path = save_pipeline(xgboost_pipeline, model_directory / "xgboost_churn_model.pkl")

    results: dict[str, Any] = {
        "logistic_regression": {
            "metrics": logistic_metrics,
            "model_path": str(logistic_path),
        },
        "xgboost": {
            "metrics": xgboost_metrics,
            "best_params": best_params,
            "model_path": str(xgboost_path),
        },
    }

    metrics_path = model_directory / "training_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(description="Train Telco churn prediction models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""

    args = parse_args()
    results = train_models(
        data_path=args.data_path,
        model_dir=args.model_dir,
        n_trials=args.n_trials,
        random_state=args.random_state,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
