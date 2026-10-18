import joblib
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.features import extract_advanced_physicochemical_features
except ModuleNotFoundError:
    from features import extract_advanced_physicochemical_features

class SolubilityPredictor:
    def __init__(self, model_path="models/best_model.joblib"):
        model_path = Path(model_path)
        if not model_path.is_absolute():
            candidate_paths = [
                PROJECT_ROOT / model_path,
                Path.cwd() / model_path,
                Path(__file__).resolve().parent / model_path,
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    model_path = candidate
                    break

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please run train.py first.")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def predict_from_sequence(self, sequence_str):
        feats = extract_advanced_physicochemical_features(sequence_str)
        if feats is None:
            raise ValueError("Invalid amino acid sequence provided.")
            
        df_feat = pd.DataFrame([feats])
        for col in self.feature_names:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
                
        X = df_feat[self.feature_names]
        sol_pred = float(self.model.predict(X)[0])
        sol_pred = float(np.clip(sol_pred, 0.0, 100.0))
        
        return {
            "predicted_solubility_percent": round(sol_pred, 2),
            "is_soluble": bool(sol_pred >= 50.0),
            "seq_length": feats.get("seq_length", 0),
            "gravy_score": round(feats.get("gravy_score", 0.0), 3),
            "net_charge_density": round(feats.get("net_charge_density", 0.0), 3)
        }

    def predict_batch_fasta(self, fasta_content_str):
        """
        Parses multi-entry FASTA formatted string and predicts solubility for all records.
        """
        records = []
        current_header = "Unassigned"
        current_seq = []
        
        for line in fasta_content_str.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    records.append((current_header, "".join(current_seq)))
                current_header = line[1:].split()[0]  # First word after >
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_seq:
            records.append((current_header, "".join(current_seq)))
            
        if not records:
            lines = [l.strip() for l in fasta_content_str.splitlines() if l.strip()]
            records = [(f"Seq_{idx+1}", seq) for idx, seq in enumerate(lines)]
            
        results = []
        for header, seq in records:
            try:
                pred_dict = self.predict_from_sequence(seq)
                results.append({
                    "Protein_ID": header,
                    "Predicted_Solubility (%)": pred_dict["predicted_solubility_percent"],
                    "Classification": "Soluble" if pred_dict["is_soluble"] else "Insoluble",
                    "Sequence_Length": pred_dict["seq_length"],
                    "GRAVY_Score": pred_dict["gravy_score"],
                    "Net_Charge_Density": pred_dict["net_charge_density"],
                    "Sequence": seq
                })
            except Exception:
                continue
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    predictor = SolubilityPredictor()
    test_seq = "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLPSEP"
    print("Single Test:", predictor.predict_from_sequence(test_seq))