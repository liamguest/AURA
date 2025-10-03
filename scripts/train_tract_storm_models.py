"""Train and evaluate baseline + tuned regressors on the tract_storm dataset.

Algorithms covered (per midterm scope):
- KNeighborsRegressor
- DecisionTreeRegressor
- RandomForestRegressor
- BaggingRegressor (with decision tree base estimator)

Outputs:
- docs/tract_storm_model_performance.csv (tabular metrics)
- docs/tract_storm_model_performance.md (markdown summary)
- docs/tract_storm_best_params.json (best hyperparameters per model)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"

TARGET_LOG = "fema_claims_total_log1p"
TARGET_RAW = "fema_claims_total"

EXCLUDE_COLUMNS = {
    TARGET_LOG,
    TARGET_RAW,
    "fema_claims_pc",
    "population_for_pc",
    "tract_geoid",
    "disasterNumber",
    "state_abbr",
    "county_name",
    "acs_state_fips",
    "state_fips",
    "coverage_label",
}


@dataclass
class TrainConfig:
    dataset_path: Path = PROCESSED_DIR / "tract_storm_features.csv"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: Dict[str, List]


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


METRIC_FUNCS = {
    "r2": r2_score,
    "rmse": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "mae": mean_absolute_error,
    "mape": safe_mape,
    "explained_variance": explained_variance_score,
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_LOG not in df.columns:
        raise ValueError(f"Expected column '{TARGET_LOG}' to be present. Run build_tract_storm.py first.")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_cols]
    y_log = df[TARGET_LOG]
    y_raw = df[TARGET_RAW]
    return X, y_log, y_raw


def make_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)],
        remainder="drop",
    )
    return preprocessor


def get_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="knn",
            estimator=KNeighborsRegressor(),
            param_grid={
                "model__n_neighbors": [5, 10, 15, 25, 35],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeRegressor(random_state=42),
            param_grid={
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestRegressor(random_state=42, n_estimators=200),
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.75, 1.0],
            },
        ),
        ModelSpec(
            name="bagging",
            estimator=BaggingRegressor(random_state=42),
            param_grid={
                "model__n_estimators": [100, 200, 400],
                "model__max_samples": [0.5, 0.75, 1.0],
                "model__max_features": [0.5, 0.75, 1.0],
            },
        ),
    ]


def evaluate_predictions(y_true_raw: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_pred_raw = np.expm1(y_pred_log)
    metrics = {name: float(func(y_true_raw, y_pred_raw)) for name, func in METRIC_FUNCS.items()}
    return metrics


def run_baseline(
    spec: ModelSpec,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
) -> Tuple[Pipeline, Dict[str, float]]:
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", spec.estimator)])
    pipeline.fit(X_train, y_train_log)
    y_pred_log = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test_raw.to_numpy(), y_pred_log)
    return pipeline, metrics


def run_grid_search(
    spec: ModelSpec,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    config: TrainConfig,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, object]]:
    base_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", spec.estimator)])
    grid = GridSearchCV(
        base_pipeline,
        param_grid=spec.param_grid,
        cv=config.cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=config.n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train_log)
    best_pipeline = grid.best_estimator_
    y_pred_log = best_pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test_raw.to_numpy(), y_pred_log)
    best_params = grid.best_params_
    return best_pipeline, metrics, best_params


def train_and_evaluate(config: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    df = load_dataset(config.dataset_path)
    X, y_log, y_raw = build_feature_matrix(df)

    feature_names = list(X.columns)
    preprocessor = make_preprocessor(feature_names)

    X_train, X_test, y_train_log, _, _, y_test_raw = train_test_split(
        X,
        y_log,
        y_raw,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    results: List[Dict[str, object]] = []
    best_params: Dict[str, Dict[str, object]] = {}

    for spec in get_model_specs():
        _, baseline_metrics = run_baseline(
            spec,
            preprocessor,
            X_train,
            y_train_log,
            X_test,
            y_test_raw,
        )
        results.append(
            {
                "model": spec.name,
                "phase": "baseline",
                **baseline_metrics,
            }
        )

        _, tuned_metrics, tuned_params = run_grid_search(
            spec,
            preprocessor,
            X_train,
            y_train_log,
            X_test,
            y_test_raw,
            config,
        )
        results.append(
            {
                "model": spec.name,
                "phase": "tuned",
                **tuned_metrics,
            }
        )
        best_params[spec.name] = tuned_params

    results_df = pd.DataFrame(results)
    return results_df, best_params


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    rounded = df.copy()
    metric_cols = [c for c in df.columns if c not in {"model", "phase"}]
    rounded[metric_cols] = rounded[metric_cols].applymap(lambda x: round(x, 4) if pd.notna(x) else x)

    headers = list(rounded.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + " --- |" * len(headers))
    for row in rounded.itertuples(index=False):
        cells = ["" if pd.isna(val) else str(val) for val in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(results: pd.DataFrame, best_params: Dict[str, Dict[str, object]]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DOCS_DIR / "tract_storm_model_performance.csv"
    md_path = DOCS_DIR / "tract_storm_model_performance.md"
    json_path = DOCS_DIR / "tract_storm_best_params.json"

    results.to_csv(csv_path, index=False)

    md_lines = ["# tract_storm Model Performance", ""]
    md_lines.append(dataframe_to_markdown(results))
    md_path.write_text("\n".join(md_lines))

    with open(json_path, "w") as fp:
        json.dump(best_params, fp, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=TrainConfig.dataset_path)
    parser.add_argument("--test-size", type=float, default=TrainConfig.test_size)
    parser.add_argument("--random-state", type=int, default=TrainConfig.random_state)
    parser.add_argument("--cv-folds", type=int, default=TrainConfig.cv_folds)
    parser.add_argument("--n-jobs", type=int, default=TrainConfig.n_jobs)
    args = parser.parse_args()
    return TrainConfig(
        dataset_path=args.dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    config = parse_args()
    results, best_params = train_and_evaluate(config)
    write_outputs(results, best_params)


if __name__ == "__main__":
    main()
