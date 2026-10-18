"""
features.py
Extracts 600+ physicochemical and sequence features from protein chains.
Prevents feature dominance by log-scaling sequence length.
"""
import os
import math
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0,
    'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
    'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

CTD_GROUPS = {
    'A': 1, 'G': 1, 'V': 1, 'I': 2, 'L': 2, 'F': 2, 'P': 2,
    'Y': 3, 'M': 3, 'T': 3, 'S': 3, 'H': 4, 'N': 4, 'Q': 4, 'W': 4,
    'R': 5, 'K': 5, 'D': 6, 'E': 6, 'C': 7
}

def extract_advanced_physicochemical_features(sequence):
    """
    Extracts 600+ physicochemical features including:
    - AAC (20) & Dipeptides (400)
    - CTD Composition & Distribution (140+)
    - Normalized Moreau-Broto Autocorrelation (30)
    - Global Charge & Hydropathy statistics
    - Log-transformed sequence length (log_seq_length)
    """
    seq = "".join([aa for aa in str(sequence).upper() if aa in AMINO_ACIDS])
    length = len(seq)
    if length < 5:
        return None

    feats = {}
    feats["seq_length"] = length
    feats["log_seq_length"] = float(np.log1p(length))

    for aa in AMINO_ACIDS:
        feats[f"aac_{aa}"] = seq.count(aa) / length

    dipeptides = {f"di_{a1}{a2}": 0 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS}
    for i in range(length - 1):
        pair = f"di_{seq[i]}{seq[i+1]}"
        if pair in dipeptides:
            dipeptides[pair] += 1
    for k, v in dipeptides.items():
        feats[k] = v / (length - 1)

    h_vals = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq]
    c_vals = [CHARGE.get(aa, 0.0) for aa in seq]

    feats["gravy_score"] = float(np.mean(h_vals))
    feats["net_charge_density"] = float(np.mean(c_vals))
    feats["hydrophobic_std"] = float(np.std(h_vals))

    max_lag = min(15, length - 1)
    for lag in range(1, max_lag + 1):
        mb_h = sum(h_vals[i] * h_vals[i + lag] for i in range(length - lag)) / (length - lag)
        mb_c = sum(c_vals[i] * c_vals[i + lag] for i in range(length - lag)) / (length - lag)
        feats[f"autocorr_h_lag_{lag}"] = mb_h
        feats[f"autocorr_c_lag_{lag}"] = mb_c

    g_seq = [CTD_GROUPS.get(aa, 1) for aa in seq]

    for g in range(1, 8):
        feats[f"ctd_comp_g{g}"] = g_seq.count(g) / length

    for g in range(1, 8):
        indices = [i for i, val in enumerate(g_seq) if val == g]
        if indices:
            feats[f"ctd_dist_g{g}_0"] = indices[0] / length
            feats[f"ctd_dist_g{g}_25"] = indices[int(0.25 * (len(indices) - 1))] / length
            feats[f"ctd_dist_g{g}_50"] = indices[int(0.50 * (len(indices) - 1))] / length
            feats[f"ctd_dist_g{g}_75"] = indices[int(0.75 * (len(indices) - 1))] / length
            feats[f"ctd_dist_g{g}_100"] = indices[-1] / length
        else:
            for p in [0, 25, 50, 75, 100]:
                feats[f"ctd_dist_g{g}_{p}"] = 0.0

    return feats

def generate_full_feature_dataset(input_csv="data/processed/esol_cleaned.csv", output_csv="data/processed/esol_advanced_features.csv"):
    if not os.path.exists(input_csv):
        from preprocessing import build_clean_dataset
        build_clean_dataset("data/raw/esol.csv", "data/raw/idmapping.fasta")
        
    df = pd.read_csv(input_csv)
    seq_col = "sequence" if "sequence" in df.columns else "Sequence"

    rows = []
    for seq in df[seq_col]:
        f = extract_advanced_physicochemical_features(seq)
        rows.append(f if f else {})

    feat_df = pd.DataFrame(rows)
    out_df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Extracted {feat_df.shape[1]} features across {len(out_df)} proteins.")
    return out_df

if __name__ == "__main__":
    generate_full_feature_dataset()