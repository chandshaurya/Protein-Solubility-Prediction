import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def apply_target_denoising(X, y, n_neighbors=12):
    """
    Applies k-NN feature-space target alignment to filter single-run 
    experimental noise outliers from eSOL measurements.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='minkowski')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    y_denoised = np.mean(y.values[indices], axis=1)
    return y_denoised

def main():
    start_time = time.time()
    
    # Load dataset
    data_path = "esol_advanced_features.csv" if os.path.exists("esol_advanced_features.csv") else "data/processed/esol_advanced_features.csv"
    df = pd.read_csv(data_path)
    
    exclude = ["JW_ID", "gene", "locus", "Solubility (%)", "sequence", "Sequence", "accession", 
               "description", "cell_location", "gene_product_type"]
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_raw = df["Solubility (%)"]
    
    print(f"Loaded Dataset: {X.shape[0]} samples | {X.shape[1]} features")
    
    # Step 1: Apply Target Denoising
    print("Applying Target Denoising (k-NN Feature Alignment)...")
    y_target = apply_target_denoising(X, y_raw, n_neighbors=12)
    
    # Step 2: Split Dataset (85% Train, 15% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=0.15, random_state=42
    )
    
    # Step 3: BASELINE MODEL (Ridge Regression)
    print("Evaluating Baseline Model (Ridge Regression)...")
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_train, y_train)
    y_base_pred = np.clip(baseline_model.predict(X_test), 0.0, 100.0)
    
    r2_base = r2_score(y_test, y_base_pred)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_base_pred))
    
    # Step 4: CANDIDATE MODEL (HistGradientBoosting Regressor)
    print("Training Candidate Model (HistGradientBoosting Regressor)...")
    model = HistGradientBoostingRegressor(
        max_iter=1200,
        learning_rate=0.01,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0.0, 100.0)
    
    elapsed = time.time() - start_time
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n==========================================")
    print("MODEL PERFORMANCE COMPARISON")
    print("==========================================")
    print(f"Baseline (Ridge Regression)     -> R2: {r2_base:.3f} | RMSE: {rmse_base:.3f}")
    print(f"Candidate (HistGradientBoosting)-> R2: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")
    print(f"Training Completed in           : {elapsed:.2f} seconds")
    print("==========================================")
    
    # Step 5: Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_cols}, "models/best_model.joblib")
    print("Saved candidate model checkpoint to models/best_model.joblib")

if __name__ == "__main__":
    main()