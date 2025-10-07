"""Generate visualizations for model performance and bias analysis.

Creates:
- Comparative performance bar charts
- Actual vs predicted scatter plots for best models
- Residual analysis by vulnerability quartiles (bias diagnostics)
- Feature importance plots (for tree-based models)

Outputs saved to: midterm_MVP/visuals/model_results/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

REPO_ROOT = Path(__file__).resolve().parents[1]
AURA_ROOT = REPO_ROOT.parent
PROCESSED_DIR = AURA_ROOT / "data" / "processed"
DOCS_DIR = AURA_ROOT / "docs"
VISUALS_DIR = REPO_ROOT / "visuals" / "model_results"

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
}

VULNERABILITY_COLS = {
    "pct_poverty": "Poverty Rate",
    "pct_hispanic": "Hispanic %",
    "pct_black": "Black %",
    "pct_limited_english": "Limited English %",
}


def load_data():
    """Load dataset and performance results."""
    df = pd.read_csv(PROCESSED_DIR / "tract_storm_features.csv")
    results = pd.read_csv(DOCS_DIR / "tract_storm_model_performance.csv")

    with open(DOCS_DIR / "tract_storm_best_params.json", "r") as f:
        best_params = json.load(f)

    return df, results, best_params


def prepare_features(df):
    """Prepare feature matrix and target."""
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    X = df[feature_cols]

    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    y_log = df[TARGET_LOG]
    y_raw = df[TARGET_RAW]
    return X, y_log, y_raw, numeric_cols


def make_preprocessor(feature_names):
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


def train_best_models(X_train, y_train_log, X_test, y_test_log, y_test_raw, feature_names, best_params):
    """Train best models and generate predictions."""
    preprocessor = make_preprocessor(feature_names)

    models = {
        "Decision Tree": Pipeline([
            ("preprocess", preprocessor),
            ("model", DecisionTreeRegressor(
                random_state=42,
                max_depth=best_params["decision_tree"]["model__max_depth"],
                min_samples_split=best_params["decision_tree"]["model__min_samples_split"],
                min_samples_leaf=best_params["decision_tree"]["model__min_samples_leaf"],
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                random_state=42,
                n_estimators=best_params["random_forest"]["model__n_estimators"],
                max_depth=best_params["random_forest"]["model__max_depth"],
                min_samples_split=best_params["random_forest"]["model__min_samples_split"],
                min_samples_leaf=best_params["random_forest"]["model__min_samples_leaf"],
                max_features=best_params["random_forest"]["model__max_features"],
            ))
        ]),
        "Bagging": Pipeline([
            ("preprocess", preprocessor),
            ("model", BaggingRegressor(
                random_state=42,
                n_estimators=best_params["bagging"]["model__n_estimators"],
                max_samples=best_params["bagging"]["model__max_samples"],
                max_features=best_params["bagging"]["model__max_features"],
            ))
        ]),
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        y_pred_raw = np.expm1(y_pred_log)
        predictions[name] = {
            "model": model,
            "y_pred_raw": y_pred_raw,
            "y_pred_log": y_pred_log,
        }

    return predictions


def plot_performance_comparison(results):
    """Create bar chart comparing model performance."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Model Performance Comparison: Baseline vs Tuned", fontsize=16, fontweight='bold')

    metrics = ["r2", "rmse", "mae", "mape", "explained_variance"]
    metric_labels = ["R² Score", "RMSE", "MAE", "MAPE", "Explained Variance"]

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]

        pivot = results.pivot(index="model", columns="phase", values=metric)
        pivot = pivot.reindex(["knn", "decision_tree", "random_forest", "bagging"])

        x = np.arange(len(pivot))
        width = 0.35

        ax.bar(x - width/2, pivot["baseline"], width, label="Baseline", alpha=0.8)
        ax.bar(x + width/2, pivot["tuned"], width, label="Tuned", alpha=0.8)

        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(["KNN", "Decision Tree", "Random Forest", "Bagging"], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test_raw, predictions):
    """Create actual vs predicted scatter plots for best models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Actual vs Predicted FEMA Claims (Best Models)", fontsize=16, fontweight='bold')

    for idx, (name, pred_dict) in enumerate(predictions.items()):
        ax = axes[idx]
        y_pred = pred_dict["y_pred_raw"]

        # Scatter plot
        ax.scatter(y_test_raw, y_pred, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_test_raw.min(), y_pred.min())
        max_val = max(y_test_raw.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel("Actual Claims ($)")
        ax.set_ylabel("Predicted Claims ($)")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)

        # Add R² annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test_raw, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_residuals_by_vulnerability(y_test_raw, predictions, X_test, df_full):
    """Analyze residuals by vulnerability quartiles."""
    # Get vulnerability data for test set
    test_indices = X_test.index
    vuln_data = df_full.loc[test_indices, list(VULNERABILITY_COLS.keys())].copy()

    fig, axes = plt.subplots(len(VULNERABILITY_COLS), 3, figsize=(18, 16))
    fig.suptitle("Residual Analysis by Vulnerability Quartiles (Bias Diagnostics)",
                 fontsize=16, fontweight='bold')

    for model_idx, (model_name, pred_dict) in enumerate(predictions.items()):
        y_pred = pred_dict["y_pred_raw"]
        residuals = y_test_raw.values - y_pred

        for vuln_idx, (col, label) in enumerate(VULNERABILITY_COLS.items()):
            ax = axes[vuln_idx, model_idx]

            # Create quartiles
            vuln_values = vuln_data[col].fillna(0)
            quartiles = pd.qcut(vuln_values, q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates='drop')

            # Box plot of residuals by quartile
            data_for_plot = pd.DataFrame({"Residual": residuals, "Quartile": quartiles})
            data_for_plot.boxplot(column="Residual", by="Quartile", ax=ax)

            ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_xlabel(f"{label} Quartile")
            ax.set_ylabel("Residual ($)")
            ax.set_title(f"{model_name}\n{label}")
            plt.sca(ax)
            plt.xticks(rotation=45)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(predictions, feature_names):
    """Plot feature importance for tree-based models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Feature Importance (Top 15 Features)", fontsize=16, fontweight='bold')

    for idx, (name, pred_dict) in enumerate(predictions.items()):
        ax = axes[idx]
        model = pred_dict["model"]

        # Extract the actual model from pipeline
        actual_model = model.named_steps["model"]

        if hasattr(actual_model, "feature_importances_"):
            importances = actual_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            ax.barh(range(15), importances[indices])
            ax.set_yticks(range(15))
            ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel("Importance")
            ax.set_title(name)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, results, best_params = load_data()

    print("Preparing features...")
    X, y_log, y_raw, feature_names = prepare_features(df)

    print("Splitting data...")
    X_train, X_test, y_train_log, y_test_log, _, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    print("Training best models...")
    predictions = train_best_models(X_train, y_train_log, X_test, y_test_log, y_test_raw, feature_names, best_params)

    print("Generating visualizations...")

    # 1. Performance comparison
    fig1 = plot_performance_comparison(results)
    fig1.savefig(VISUALS_DIR / "01_performance_comparison.png", bbox_inches='tight')
    print(f"✓ Saved: 01_performance_comparison.png")

    # 2. Actual vs predicted
    fig2 = plot_actual_vs_predicted(y_test_raw, predictions)
    fig2.savefig(VISUALS_DIR / "02_actual_vs_predicted.png", bbox_inches='tight')
    print(f"✓ Saved: 02_actual_vs_predicted.png")

    # 3. Residuals by vulnerability
    fig3 = plot_residuals_by_vulnerability(y_test_raw, predictions, X_test, df)
    fig3.savefig(VISUALS_DIR / "03_residuals_by_vulnerability.png", bbox_inches='tight')
    print(f"✓ Saved: 03_residuals_by_vulnerability.png")

    # 4. Feature importance
    fig4 = plot_feature_importance(predictions, feature_names)
    fig4.savefig(VISUALS_DIR / "04_feature_importance.png", bbox_inches='tight')
    print(f"✓ Saved: 04_feature_importance.png")

    print(f"\n✅ All visualizations saved to: {VISUALS_DIR}")
    plt.close('all')


if __name__ == "__main__":
    main()
