from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAP_BACKGROUND_SIZE = 20
SHAP_SAMPLE_SIZE = 25


def build_experiments(feature_columns):
    feature_columns = list(feature_columns)
    feature_set = set(feature_columns)
    reported_features = [
        col for col in ("mw_kda_reported", "pi_reported") if col in feature_set
    ]
    sequence_summary_features = [
        col for col in (
            "seq_length",
            "log_seq_length",
            "gravy_score",
            "net_charge_density",
            "hydrophobic_std",
        ) if col in feature_set
    ]
    aac_features = [col for col in feature_columns if col.startswith("aac_")]
    ablation_groups = {
        "E5": reported_features,
        "E6": sequence_summary_features,
        "E7": aac_features,
    }
    return {
        "E0": [],
        "E1": reported_features,
        "E2": sequence_summary_features,
        "E3": aac_features,
        "E4": feature_columns,
        "E5": [col for col in feature_columns if col not in ablation_groups["E5"]],
        "E6": [col for col in feature_columns if col not in ablation_groups["E6"]],
        "E7": [col for col in feature_columns if col not in ablation_groups["E7"]],
    }


def _build_experiments():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "esol_advanced_features.csv"
    if dataset_path.exists():
        sample = pd.read_csv(dataset_path, nrows=1)
        exclude = {
            "JW_ID", "gene", "locus", "Solubility (%)", "sequence", "Sequence",
            "accession", "description", "cell_location", "gene_product_type",
        }
        feature_columns = [
            col for col in sample.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]
    else:
        feature_columns = [
            "mw_kda_reported", "pi_reported", "seq_length", "log_seq_length",
            "gravy_score", "net_charge_density", "hydrophobic_std", "aac_A",
        ]
    return build_experiments(feature_columns)


EXPERIMENTS = _build_experiments()


def apply_target_denoising(X, y, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return np.mean(y.values[indices], axis=1)


def bootstrap_rmse_interval(y_true, y_pred, n_bootstrap=1000, random_state=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same non-zero shape.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    errors = np.square(y_true - y_pred)
    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(
        0, errors.size, size=(n_bootstrap, errors.size)
    )
    bootstrap_rmse = np.sqrt(errors[sample_indices].mean(axis=1))
    lower, upper = np.quantile(bootstrap_rmse, [0.025, 0.975])
    return float(lower), float(upper)


def run_feature_ablation(df, feature_names, model, test_size=0.15, random_state=42):
    experiments = build_experiments(feature_names)
    if not experiments["E1"] or not experiments["E2"] or not experiments["E3"]:
        raise ValueError("Dataset is missing one or more required ablation feature groups.")

    X = df[feature_names].apply(pd.to_numeric, errors="raise")
    y = df["Solubility (%)"].to_numpy()
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state
    )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    predictions = {
        "E0": np.full(y_test.shape, y_train.mean(), dtype=float),
    }
    for experiment_name in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        selected_features = experiments[experiment_name]
        estimator = clone(model)
        estimator.fit(X_train[selected_features], y_train)
        predictions[experiment_name] = np.clip(
            estimator.predict(X_test[selected_features]), 0.0, 100.0
        )

    full_rmse = np.sqrt(mean_squared_error(y_test, predictions["E4"]))
    rows = []
    for experiment_name, y_pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        lower, upper = bootstrap_rmse_interval(
            y_test, y_pred, random_state=random_state
        )
        rows.append({
            "Experiment": experiment_name,
            "Feature_Group": {
                "E0": "Mean-only baseline",
                "E1": "Reported molecular properties",
                "E2": "Sequence summary properties",
                "E3": "Amino-acid composition",
                "E4": "All available features",
                "E5": "All except reported properties",
                "E6": "All except sequence summaries",
                "E7": "All except amino-acid composition",
            }[experiment_name],
            "Feature_Count": len(experiments[experiment_name]),
            "R2": r2_score(y_test, y_pred),
            "RMSE": rmse,
            "RMSE_CI_95_Lower": lower,
            "RMSE_CI_95_Upper": upper,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE_Delta_vs_All": rmse - full_rmse,
        })
    return pd.DataFrame(rows)


def _save_ablation_plot(ablation, figures_dir):
    ordered = ablation.sort_values("RMSE", ascending=True)
    lower_error = ordered["RMSE"] - ordered["RMSE_CI_95_Lower"]
    upper_error = ordered["RMSE_CI_95_Upper"] - ordered["RMSE"]

    plt.figure(figsize=(8, 4.5))
    plt.barh(ordered["Feature_Group"], ordered["RMSE"], color="teal")
    plt.errorbar(
        ordered["RMSE"],
        ordered["Feature_Group"],
        xerr=[lower_error, upper_error],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    plt.xlabel("Test RMSE (percentage points; 95% bootstrap interval)")
    plt.title("Feature-family ablation on the raw-solubility holdout")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_ablation_rmse.png", dpi=300)
    plt.close()


def run_depth_analysis():
    print("Running In-Depth Error, Feature Importance & Biological Analysis...")
    
    data_path = PROJECT_ROOT / "data" / "processed" / "esol_advanced_features.csv"
    df = pd.read_csv(data_path)
    
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Run train.py first.")
        
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    exclude = ["JW_ID", "gene", "locus", "Solubility (%)", "sequence", "Sequence", "accession", 
               "description", "cell_location", "gene_product_type"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_raw = df["Solubility (%)"]
    
    y_target = apply_target_denoising(X, y_raw, n_neighbors=12)
    
    # Train/Test Split matching train.py & evaluate.py
    X_train, X_test, _, y_test = train_test_split(
        X, y_target, test_size=0.15, random_state=42
    )
    
    df_test = df.iloc[X_test.index].copy()
    y_test_pred = np.clip(model.predict(X_test), 0.0, 100.0)
    
    df_test["Consensus_Solubility"] = y_test
    df_test["Predicted_Solubility"] = y_test_pred
    df_test["Absolute_Error"] = np.abs(y_test - y_test_pred)
    
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use measured targets and train-only imputation for a leakage-conscious ablation.
    print("Running raw-target feature-family ablation...")
    ablation = run_feature_ablation(df, feature_names, model)
    ablation.to_csv(tables_dir / "feature_ablation.csv", index=False)
    _save_ablation_plot(ablation, figures_dir)
    print("Saved feature-family ablation table and RMSE plot.")
    
    # 1. Top High Error Cases
    top_errors = df_test.sort_values(by="Absolute_Error", ascending=False).head(10)
    cols_to_show = [c for c in ["gene", "locus", "Solubility (%)", "Consensus_Solubility", "Predicted_Solubility", "Absolute_Error"] if c in df_test.columns]
    top_errors[cols_to_show].to_csv(tables_dir / "top_10_high_error_cases.csv", index=False)
    print("Saved High-Error Test Cases to results/tables/top_10_high_error_cases.csv")
    
    # 2. Pre-compute and Save Feature Importance Image
    print("Computing Permutation Feature Importances...")
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances_mean = np.asarray(result["importances_mean"])
    sorted_importances_idx = importances_mean.argsort()[::-1][:12]

    top_features = np.array(feature_names)[sorted_importances_idx]
    top_scores = importances_mean[sorted_importances_idx]
    
    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        x=top_scores,
        y=top_features,
        hue=top_features,
        palette="viridis",
        legend=False,
    )
    plt.title("Top 12 Permutation Feature Importances")
    plt.xlabel("Mean Importance Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=300)
    plt.close()
    print("Saved results/figures/feature_importance.png")

    # 3. SHAP global attributions; a fixed subset keeps permutation explanations practical.
    background = X_train.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_train)),
        random_state=42,
    )
    shap_sample = X_test.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_test)),
        random_state=42,
    )
    print("Computing SHAP values...")
    explainer = shap.Explainer(
        model.predict,
        background,
        algorithm="permutation",
    )
    shap_values = explainer(
        shap_sample,
        max_evals=2 * len(feature_names) + 1,
    )
    shap.plots.beeswarm(
        shap_values,
        max_display=12,
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved results/figures/shap_summary_plot.png")
    
    # 4. GRAVY Stratification Plot
    if "gravy_score" in df_test.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df_test, 
            x="gravy_score", 
            y="Consensus_Solubility", 
            hue="Predicted_Solubility", 
            palette="mako", 
            alpha=0.7
        )
        plt.axvline(0, color="red", linestyle="--", label="Hydropathy Neutral Line (0.0)")
        plt.title("GRAVY Hydropathy vs Consensus Solubility (Test Set)")
        plt.xlabel("GRAVY Score (Hydrophobicity)")
        plt.ylabel("Consensus Solubility (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(figures_dir / "gravy_vs_solubility.png", dpi=300)
        plt.close()
        print("Saved results/figures/gravy_vs_solubility.png")
        
    print("Depth analysis completed successfully!")

if __name__ == "__main__":
    run_depth_analysis()