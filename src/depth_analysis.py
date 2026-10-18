from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHAP_BACKGROUND_SIZE = 20
SHAP_SAMPLE_SIZE = 25


def _build_experiments():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "esol_advanced_features.csv"
    feature_columns = []
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        exclude = {"JW_ID", "gene", "locus", "Solubility (%)", "sequence", "Sequence", "accession",
                   "description", "cell_location", "gene_product_type", "gene_product_type"}
        feature_columns = [
            col for col in df.columns if col not in exclude and isinstance(col, str)
        ]
    else:
        feature_columns = [
            "mw_kda_reported", "pi_reported", "length", "gravy", "aromaticity",
            "instability_index", "aac_A", "aac_C", "aac_D", "aac_E", "aac_F",
            "aac_G", "aac_H", "aac_I", "aac_K", "aac_L", "aac_M", "aac_N",
            "aac_P", "aac_Q", "aac_R", "aac_S", "aac_T", "aac_V", "aac_W",
            "aac_Y"
        ]

    aac_features = [col for col in feature_columns if str(col).startswith("aac_")]
    return {
        "E0": [],
        "E1": ["mw_kda_reported", "pi_reported"],
        "E2": ["length", "gravy", "aromaticity", "instability_index"],
        "E3": aac_features if aac_features else ["aac_A", "aac_C"],
        "E4": feature_columns,
    }


EXPERIMENTS = _build_experiments()

def apply_target_denoising(X, y, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return np.mean(y.values[indices], axis=1)

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