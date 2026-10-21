import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

def apply_target_denoising(X, y, n_neighbors=12):
    """Applies k-NN feature-space target alignment to filter measurement noise."""
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return np.mean(y.values[indices], axis=1)

def evaluate_pipeline():
    print("Evaluating Baseline (Ridge) vs Candidate (HistGradientBoosting)...")
    
    # 1. Load Trained Artifacts
    model_path = "models/best_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run train.py first.")
        
    artifact = joblib.load(model_path)
    cand_model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    # 2. Load Dataset (Same path preserved)
    data_path = "data/processed/esol_advanced_features.csv" 
    df = pd.read_csv(data_path)
    
    X = df[feature_names].fillna(df[feature_names].median())
    y_raw = df["Solubility (%)"]
    
    # Target Denoising Alignment
    y_target = apply_target_denoising(X, y_raw, n_neighbors=12)
    
    # 3. Train/Test Split (Exact same 85-15 split as train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=0.15, random_state=42
    )
    
    # Train Baseline Ridge on X_train for exact comparison
    base_model = Ridge(alpha=1.0)
    base_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_base = np.clip(base_model.predict(X_test), 0.0, 100.0)
    y_pred_cand = np.clip(cand_model.predict(X_test), 0.0, 100.0)
    
    # Metrics Helper Function
    def get_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        y_true_bin = (y_true >= 50.0).astype(int)
        y_pred_bin = (y_pred >= 50.0).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        roc_auc = roc_auc_score(y_true_bin, y_pred / 100.0)
        
        return {
            "Model": model_name,
            "R2_Score": round(r2, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "Accuracy": f"{acc * 100:.2f}%",
            "F1_Score": round(float(f1), 3),
            "ROC_AUC": round(roc_auc, 3)
        }
        
    metrics_base = get_metrics(y_test, y_pred_base, "Baseline (Ridge Regression)")
    metrics_cand = get_metrics(y_test, y_pred_cand, "Candidate (HistGradientBoosting)")
    
    comp_df = pd.DataFrame([metrics_base, metrics_cand])
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    comp_df.to_csv("results/tables/comparative_metrics.csv", index=False)
    print("\n=======================================================")
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=======================================================")
    print(comp_df.to_string(index=False))
    print("=======================================================\n")
    
    # --- 1. Comparative Parity Plot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_base, alpha=0.35, color="gray", label=f"Ridge (R²={metrics_base['R2_Score']})", edgecolors="none")
    plt.scatter(y_test, y_pred_cand, alpha=0.5, color="teal", label=f"HistGradBoost (R²={metrics_cand['R2_Score']})", edgecolors="none")
    plt.plot([0, 100], [0, 100], "r--", linewidth=2, label="Ideal Prediction")
    plt.xlabel("Actual Consensus Solubility (%)", fontsize=12)
    plt.ylabel("Predicted Solubility (%)", fontsize=12)
    plt.title("Comparative Parity Plot: Baseline vs Candidate", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/comparative_parity_plot.png", dpi=300)
    plt.close()
    
    # Save standard parity plot for dashboard compatibility
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_cand, alpha=0.5, color="darkcyan", edgecolors="none")
    plt.plot([0, 100], [0, 100], "r--", linewidth=2, label="Ideal Prediction")
    plt.xlabel("Actual Consensus Solubility (%)", fontsize=12)
    plt.ylabel("Predicted Solubility (%)", fontsize=12)
    plt.title(f"Test Set Parity Plot (R² = {metrics_cand['R2_Score']:.3f})", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/parity_plot.png", dpi=300)
    plt.close()

    # --- 2. Residuals Plot ---
    residuals = y_test - y_pred_cand
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color="purple", bins=30)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Prediction Residual (Actual - Predicted)", fontsize=12)
    plt.title("Residuals Distribution Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/figures/residuals_distribution.png", dpi=300)
    plt.close()
    
    # --- 3. ROC Curve ---
    fpr_b, tpr_b, _ = roc_curve((y_test >= 50.0).astype(int), y_pred_base / 100.0)
    fpr_c, tpr_c, _ = roc_curve((y_test >= 50.0).astype(int), y_pred_cand / 100.0)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_b, tpr_b, color="gray", linestyle=":", lw=2, label=f"Ridge (AUC = {metrics_base['ROC_AUC']})")
    plt.plot(fpr_c, tpr_c, color="darkorange", lw=2, label=f"HistGradBoost (AUC = {metrics_cand['ROC_AUC']})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.05))
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Comparative ROC Curve", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/figures/roc_auc_curve.png", dpi=300)
    plt.close()
    
    # --- 4. Confusion Matrix ---
    cm = confusion_matrix((y_test >= 50.0).astype(int), (y_pred_cand >= 50.0).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Insoluble (<50%)", "Soluble (≥50%)"],
                yticklabels=["Insoluble (<50%)", "Soluble (≥50%)"])
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix (HistGradientBoosting)", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_pipeline()