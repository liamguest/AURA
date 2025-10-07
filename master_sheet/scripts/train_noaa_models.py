"""Train models with NOAA features and compare to baseline.

Two analyses:
- Analysis A: 401 LA Ida tracts (100% NOAA coverage)
- Analysis B: 811 All Ida tracts (49% NOAA coverage)

Baseline for comparison: Original 5,668-row model (R² = 0.927)
"""

from __future__ import annotations

import json
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

# Paths
MASTER_SHEET_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MASTER_SHEET_DIR / "data"
DOCS_DIR = MASTER_SHEET_DIR / "docs"

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


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-6
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


METRIC_FUNCS = {
    "r2": r2_score,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "mape": safe_mape,
    "explained_variance": explained_variance_score,
}


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load one of the analysis datasets."""
    path = DATA_DIR / dataset_name
    df = pd.read_csv(path)
    print(f"Loaded {dataset_name}: {len(df)} rows")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Prepare feature matrix and target."""
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_cols]

    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    y_log = df[TARGET_LOG]
    y_raw = df[TARGET_RAW]

    return X, y_log, y_raw, numeric_cols


def make_preprocessor(feature_names: List[str]) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)],
        remainder="drop",
    )


def get_model_specs() -> List[Dict]:
    """Get model specifications with hyperparameter grids."""
    return [
        {
            "name": "knn",
            "estimator": KNeighborsRegressor(),
            "param_grid": {
                "model__n_neighbors": [5, 10, 15, 25],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        {
            "name": "decision_tree",
            "estimator": DecisionTreeRegressor(random_state=42),
            "param_grid": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
            },
        },
        {
            "name": "random_forest",
            "estimator": RandomForestRegressor(random_state=42, n_estimators=200),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.75, 1.0],
            },
        },
        {
            "name": "bagging",
            "estimator": BaggingRegressor(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 200, 400],
                "model__max_samples": [0.5, 0.75, 1.0],
                "model__max_features": [0.5, 0.75, 1.0],
            },
        },
    ]


def evaluate_predictions(y_true_raw: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions on raw scale."""
    y_pred_raw = np.expm1(y_pred_log)
    return {name: float(func(y_true_raw, y_pred_raw)) for name, func in METRIC_FUNCS.items()}


def run_grid_search(
    spec: Dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    cv_folds: int = 5,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, object]]:
    """Run grid search and evaluate."""
    base_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", spec["estimator"])])

    grid = GridSearchCV(
        base_pipeline,
        param_grid=spec["param_grid"],
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train_log)
    best_pipeline = grid.best_estimator_
    y_pred_log = best_pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test_raw.to_numpy(), y_pred_log)

    return best_pipeline, metrics, grid.best_params_


def extract_feature_importance(model: Pipeline, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance if available."""
    actual_model = model.named_steps["model"]

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
        return dict(zip(feature_names, importances))

    return {}


def train_and_evaluate(dataset_name: str, analysis_label: str) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Train models on given dataset."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {analysis_label}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # Load and prepare
    df = load_dataset(dataset_name)
    X, y_log, y_raw, feature_names = prepare_features(df)

    print(f"Features: {len(feature_names)}")
    noaa_features = [f for f in feature_names if f.startswith('noaa_')]
    print(f"NOAA features: {len(noaa_features)} → {noaa_features}")

    preprocessor = make_preprocessor(feature_names)

    # Train/test split
    X_train, X_test, y_train_log, _, _, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    results = []
    best_params = {}
    feature_importances = {}

    for spec in get_model_specs():
        print(f"\n  Training {spec['name']}...")

        try:
            best_pipeline, metrics, params = run_grid_search(
                spec, preprocessor, X_train, y_train_log, X_test, y_test_raw
            )

            results.append({"model": spec["name"], **metrics})
            best_params[spec["name"]] = params

            # Extract feature importance
            importances = extract_feature_importance(best_pipeline, feature_names)
            if importances:
                feature_importances[spec["name"]] = importances

            print(f"    R² = {metrics['r2']:.4f}, RMSE = ${metrics['rmse']:.0f}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")

    results_df = pd.DataFrame(results)

    return results_df, best_params, feature_importances


def compare_to_baseline(results_df: pd.DataFrame, analysis_label: str):
    """Compare results to baseline."""
    print(f"\n{'='*70}")
    print(f"COMPARISON TO BASELINE: {analysis_label}")
    print(f"{'='*70}")

    # Baseline from midterm (5,668 rows, no NOAA)
    baseline = {
        "knn": {"r2": 0.703, "rmse": 1997, "mae": 557},
        "decision_tree": {"r2": 0.844, "rmse": 1444, "mae": 338},
        "random_forest": {"r2": 0.924, "rmse": 1007, "mae": 266},
        "bagging": {"r2": 0.927, "rmse": 990, "mae": 262},
    }

    print(f"\n{'Model':<20} {'Baseline R²':>12} {'New R²':>12} {'Δ R²':>12} {'Baseline RMSE':>15} {'New RMSE':>12} {'Δ RMSE':>12}")
    print("-" * 110)

    for _, row in results_df.iterrows():
        model = row["model"]
        base_r2 = baseline[model]["r2"]
        new_r2 = row["r2"]
        delta_r2 = new_r2 - base_r2

        base_rmse = baseline[model]["rmse"]
        new_rmse = row["rmse"]
        delta_rmse = new_rmse - base_rmse

        print(f"{model:<20} {base_r2:>12.4f} {new_r2:>12.4f} {delta_r2:>+12.4f} {base_rmse:>15.0f} {new_rmse:>12.0f} {delta_rmse:>+12.0f}")


def save_results(
    results_df: pd.DataFrame,
    best_params: Dict,
    feature_importances: Dict,
    analysis_label: str,
    suffix: str,
):
    """Save results to disk."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Results CSV
    csv_path = DOCS_DIR / f"model_performance_{suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # Best params JSON
    json_path = DOCS_DIR / f"best_params_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Feature importances JSON
    if feature_importances:
        fi_path = DOCS_DIR / f"feature_importances_{suffix}.json"
        with open(fi_path, "w") as f:
            json.dump(feature_importances, f, indent=2)
        print(f"✓ Saved: {fi_path}")


def main():
    """Run both analyses."""
    print("="*70)
    print("NOAA FEATURE IMPACT ANALYSIS")
    print("="*70)

    # Analysis A: Louisiana Ida only (100% NOAA coverage)
    results_a, params_a, fi_a = train_and_evaluate(
        "ida_la_complete.csv",
        "Analysis A: Louisiana Ida Only (100% NOAA coverage)"
    )
    compare_to_baseline(results_a, "Analysis A")
    save_results(results_a, params_a, fi_a, "Analysis A", "ida_la_complete")

    # Analysis B: All Ida (49% NOAA coverage)
    results_b, params_b, fi_b = train_and_evaluate(
        "ida_all_partial.csv",
        "Analysis B: All Ida Tracts (49% NOAA coverage)"
    )
    compare_to_baseline(results_b, "Analysis B")
    save_results(results_b, params_b, fi_b, "Analysis B", "ida_all_partial")

    print("\n" + "="*70)
    print("✅ NOAA ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
