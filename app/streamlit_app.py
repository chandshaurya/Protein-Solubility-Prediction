import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Ensure project root directory is in Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.predict import SolubilityPredictor
except ModuleNotFoundError:
    from predict import SolubilityPredictor

st.set_page_config(
    page_title="Protein Solubility ML Platform",
    layout="wide",
    page_icon="🧬"
)

st.title("🧬 Protein Solubility Prediction & Analytics Platform")
st.markdown("Predict protein solubility percentage from primary sequence attributes and explore *E. coli* eSOL dataset analytics.")

@st.cache_resource
def load_predictor():
    model_path = "models/best_model.joblib"
    if os.path.exists(model_path):
        return SolubilityPredictor(model_path)
    return None

predictor = load_predictor()

# Sidebar Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select View", [
    "Sequence Predictor", 
    "Batch FASTA Predictor", 
    "Dataset Explorer", 
    "Evaluation & Depth Analysis"
])

# ==========================================
# 1. SINGLE SEQUENCE PREDICTOR VIEW
# ==========================================
if menu == "Sequence Predictor":
    st.subheader("Single Sequence Inference")
    st.markdown("Enter a FASTA or raw amino acid sequence to compute sequence physicochemical attributes and predict solubility.")
    
    default_seq = "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLPSEP"
    seq_input = st.text_area("Input Amino Acid Sequence (Single-letter code):", value=default_seq, height=140)
    
    if st.button("Predict Solubility", type="primary"):
        if predictor is None:
            st.error("Model checkpoint not found. Please run `python src/train.py` first to generate `models/best_model.joblib`.")
        else:
            try:
                res = predictor.predict_from_sequence(seq_input)
                
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Solubility", f"{res['predicted_solubility_percent']}%")
                col2.metric("Classification", "Soluble" if res['is_soluble'] else "Insoluble")
                col3.metric("Sequence Length", f"{res['seq_length']} AA")
                col4.metric("GRAVY Score", res['gravy_score'])
                
                st.success("Sequence prediction generated successfully!")
            except Exception as e:
                st.error(f"Error during sequence prediction: {e}")

# ==========================================
# 2. BATCH FASTA PREDICTOR VIEW
# ==========================================
elif menu == "Batch FASTA Predictor":
    st.subheader("📁 Batch FASTA Prediction & Export Engine")
    st.markdown("Upload a multi-entry FASTA file (`.fasta`, `.fa`, `.txt`) or paste raw FASTA text to perform high-throughput solubility predictions.")
    
    col_up1, col_up2 = st.columns([1, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa", "txt"])
    with col_up2:
        pasted_fasta = st.text_area("Or Paste Raw FASTA Text:", height=150, placeholder=">Protein_1\nMTDKLTSLRQYTTV...\n>Protein_2\nMAKQDYYEILGVSK...")
        
    fasta_content = None
    if uploaded_file is not None:
        fasta_content = uploaded_file.getvalue().decode("utf-8")
    elif pasted_fasta.strip():
        fasta_content = pasted_fasta.strip()
        
    if st.button("Process Batch Predictions", type="primary"):
        if predictor is None:
            st.error("Model checkpoint not found. Please run `python src/train.py` first.")
        elif not fasta_content:
            st.warning("Please upload a FASTA file or paste FASTA text to proceed.")
        else:
            with st.spinner("Extracting multi-scale physicochemical features and making predictions..."):
                try:
                    batch_df = predictor.predict_batch_fasta(fasta_content)
                    
                    if batch_df.empty:
                        st.error("No valid protein sequences could be parsed from the provided input.")
                    else:
                        st.success(f"Successfully processed **{len(batch_df)}** protein records!")
                        
                        # Summary Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Processed", f"{len(batch_df)}")
                        c2.metric("Soluble (≥50%)", f"{(batch_df['Classification'] == 'Soluble').sum()}")
                        c3.metric("Insoluble (<50%)", f"{(batch_df['Classification'] == 'Insoluble').sum()}")
                        c4.metric("Mean Solubility", f"{batch_df['Predicted_Solubility (%)'].mean():.1f}%")
                        
                        # Results Table
                        st.markdown("### Prediction Results Table")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Export Options
                        st.markdown("### 📥 Download Results")
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_data,
                            file_name="protein_solubility_batch_predictions.csv",
                            mime="text/csv",
                            type="secondary"
                        )
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")

# ==========================================
# 3. DATASET EXPLORER VIEW
# ==========================================
elif menu == "Dataset Explorer":
    st.subheader("eSOL Dataset Overview")
    data_path = "data/processed/esol_advanced_features.csv" if os.path.exists("data/processed/esol_advanced_features.csv") else "esol_advanced_features.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.write(f"Total Protein Records: **{len(df)}** | Extracted Feature Columns: **{df.shape[1]}**")
        
        st.dataframe(df.head(25), use_container_width=True)
        
        st.markdown("### Biological Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Solubility", f"{df['Solubility (%)'].mean():.2f}%")
        col2.metric("Median Solubility", f"{df['Solubility (%)'].median():.2f}%")
        col3.metric("Standard Deviation", f"{df['Solubility (%)'].std():.2f}%")
        
        col4, col5 = st.columns(2)
        col4.metric("Soluble Proteins (≥50%)", f"{(df['Solubility (%)'] >= 50).sum()} ({(df['Solubility (%)'] >= 50).mean()*100:.1f}%)")
        col5.metric("Insoluble Proteins (<50%)", f"{(df['Solubility (%)'] < 50).sum()} ({(df['Solubility (%)'] < 50).mean()*100:.1f}%)")
    else:
        st.warning("Dataset file not found at `data/processed/esol_advanced_features.csv`. Please run feature extraction first.")

# ==========================================
# 4. EVALUATION & DEPTH ANALYSIS VIEW
# ==========================================
elif menu == "Evaluation & Depth Analysis":
    st.subheader("📊 Comprehensive Model Evaluation & Biological Depth Analysis")
    st.markdown("In-depth validation performance metrics, baseline vs candidate model comparisons, diagnostic plots, error case inspections, and SHAP feature explanations.")
    
    # --- 1. MODEL COMPARISON METRICS TABLE ---
    st.markdown("### 1. Model Comparative Evaluation (Baseline vs Candidate)")
    comp_metrics_path = "results/tables/comparative_metrics.csv"
    metrics_path = "results/tables/evaluation_metrics.csv"
    
    if os.path.exists(comp_metrics_path):
        c_df = pd.read_csv(comp_metrics_path)
        st.table(c_df)
    elif os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        st.table(m_df)
    else:
        st.info("Run `python src/evaluate.py` to generate the evaluation metrics summary.")
        
    st.markdown("---")
    
    # --- 2. 2x2 PERFORMANCE PLOTS GRID ---
    st.markdown("### 2. Validation Diagnostic Plots")
    col1, col2 = st.columns(2)
    
    with col1:
        comp_parity_fig = "results/figures/comparative_parity_plot.png"
        parity_fig = "results/figures/parity_plot.png"
        
        if os.path.exists(comp_parity_fig):
            st.image(comp_parity_fig, caption="Comparative Parity Plot (Baseline Ridge vs Candidate HistGradBoost)")
        elif os.path.exists(parity_fig):
            st.image(parity_fig, caption="Regression Parity Plot (Predicted vs Consensus Solubility)")
            
        roc_fig = "results/figures/roc_auc_curve.png"
        if os.path.exists(roc_fig):
            st.image(roc_fig, caption="Receiver Operating Characteristic (ROC) Curve")

    with col2:
        res_fig = "results/figures/residuals_distribution.png"
        if os.path.exists(res_fig):
            st.image(res_fig, caption="Residuals Distribution Plot (Prediction Errors)")
            
        cm_fig = "results/figures/confusion_matrix.png"
        if os.path.exists(cm_fig):
            st.image(cm_fig, caption="Solubility Binary Confusion Matrix (≥50% Threshold)")

    st.markdown("---")

    # --- 3. SHAP FEATURE IMPORTANCE & BIOLOGICAL DRIVERS ---
    st.markdown("### 3. SHAP Feature Attribution & Biological Drivers")
    col_feat1, col_feat2 = st.columns([1.2, 1])
    
    shap_fig = "results/figures/shap_summary_plot.png"
    imp_fig = "results/figures/feature_importance.png"
    
    with col_feat1:
        if os.path.exists(shap_fig):
            st.image(shap_fig, caption="SHAP Global Feature Attribution & Value Impact Plot", use_container_width=True)
        elif os.path.exists(imp_fig):
            st.image(imp_fig, caption="Top 12 Permutation Feature Importances", use_container_width=True)
        else:
            st.info("Run `python src/depth_analysis.py` to generate feature importance plots.")
            
    with col_feat2:
        st.markdown("#### Key Predictive Feature Categories")
        st.write("• **Primary Sequence Metrics (`seq_length`, `mw_kda_reported`):** Global size and molecular weight parameters governing translation rate and expression dynamics.")
        st.write("• **Charge & Isoelectric Point (`net_charge_density`, `pi_reported`):** Electrostatic repulsion preventing aggregation.")
        st.write("• **Autocorrelations (`autocorr_h_lag_*`):** Spatial periodicity of hydrophobic and charged residues along the chain.")
        st.write("• **CTD & Amino Acid Compositions (`ctd_comp_*`, `aac_*`):** Global composition fractions of specific amino acid groups.")
    
    st.markdown("---")

    # --- 4. HIGH-ERROR CASE ANALYSIS ---
    st.markdown("### 4. High-Error Case Inspection & Error Analysis")
    high_error_path = "results/tables/top_10_high_error_cases.csv"
    
    if os.path.exists(high_error_path):
        err_df = pd.read_csv(high_error_path)
        st.markdown("Top 10 proteins with highest absolute prediction errors on the holdout test set:")
        st.dataframe(err_df, use_container_width=True)
    else:
        st.info("Run `python src/depth_analysis.py` to generate the top high-error case inspection table.")
        
    # --- 5. GRAVY STRATIFICATION PLOT ---
    gravy_fig = "results/figures/gravy_vs_solubility.png"
    if os.path.exists(gravy_fig):
        st.markdown("### 5. Hydropathy (GRAVY) Stratification")
        st.image(gravy_fig, caption="GRAVY Score vs Consensus Solubility Stratification Across Predictions", use_container_width=True)