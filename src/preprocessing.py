"""
preprocessing.py
=================
Cleaning, validation and merging utilities for the eSOL solubility dataset
and the UniProt sequence FASTA file.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

KEEP_COLUMNS = [
    "JW_ID",
    "Gene name K-12",
    "Locus name K-12",
    "Solubility (%)",
    "Calculated MW (kDa)",
    "Calculated pI",
    "Type of gene product",
    "Gene product description",
    "Cell location",
]

def load_raw_esol(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    logger.info("Loaded raw eSOL sheet: %s rows, %s columns", *df.shape)
    return df

def clean_esol(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()

    n0 = len(df)
    df = df.dropna(subset=["Solubility (%)"]).copy()
    logger.info("Dropped %d rows with missing solubility label", n0 - len(df))

    df["Gene name K-12"] = df["Gene name K-12"].astype(str).str.strip()
    n0 = len(df)
    df = df.drop_duplicates(subset=["Gene name K-12"], keep="first")
    logger.info("Dropped %d duplicate-gene rows", n0 - len(df))

    df = df.rename(
        columns={
            "Gene name K-12": "gene",
            "Locus name K-12": "locus",
            "Solubility (%)": "Solubility (%)",
            "Calculated MW (kDa)": "mw_kda_reported",
            "Calculated pI": "pi_reported",
            "Type of gene product": "gene_product_type",
            "Gene product description": "description",
            "Cell location": "cell_location",
        }
    )
    return df.reset_index(drop=True)

_HEADER_RE = re.compile(r"^>\w+\|(?P<accession>[^|]+)\|(?P<entry>\S+)\s+(?P<rest>.*)$")
_GN_RE = re.compile(r"GN=(\S+)")

def parse_fasta(path: str | Path) -> Dict[str, dict]:
    path = Path(path)
    records: Dict[str, dict] = {}
    accession = entry_name = gene = None
    seq_chunks = []

    def _flush():
        if gene and seq_chunks:
            seq = "".join(seq_chunks)
            if gene not in records:
                records[gene] = {
                    "accession": accession,
                    "entry_name": entry_name,
                    "sequence": seq,
                }

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                _flush()
                seq_chunks = []
                accession = entry_name = gene = None
                m = _HEADER_RE.match(line)
                if m:
                    accession = m.group("accession")
                    entry_name = m.group("entry")
                    gm = _GN_RE.search(m.group("rest"))
                    gene = gm.group(1) if gm else None
            else:
                seq_chunks.append(line.strip())
        _flush()

    logger.info("Parsed %d FASTA records with a GN= gene tag", len(records))
    return records

def is_valid_sequence(seq: Optional[str]) -> bool:
    if not seq or not isinstance(seq, str):
        return False
    return set(seq.upper()) <= VALID_AA

def merge_sequences(esol_df: pd.DataFrame, fasta_map: Dict[str, dict]) -> pd.DataFrame:
    df = esol_df.copy()
    df["sequence"] = df["gene"].map(lambda g: fasta_map.get(g, {}).get("sequence"))
    df["accession"] = df["gene"].map(lambda g: fasta_map.get(g, {}).get("accession"))

    df["seq_valid"] = df["sequence"].apply(is_valid_sequence)
    df = df[df["seq_valid"]].drop(columns="seq_valid").reset_index(drop=True)
    df["seq_length"] = df["sequence"].str.len()
    return df

def build_clean_dataset(raw_csv: str | Path, fasta_path: str | Path) -> pd.DataFrame:
    raw = load_raw_esol(raw_csv)
    clean = clean_esol(raw)
    
    # Save base cleaned dataset in root and data directories
    clean.to_csv("data/processed/desol_cleaned.csv", index=False)
    
    if Path(fasta_path).exists():
        fasta_map = parse_fasta(fasta_path)
        merged = merge_sequences(clean, fasta_map)
        merged.to_csv("data/processed/esol_cleaned.csv", index=False)
        return merged
        
    return clean

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data/raw/esol.csv" if (root / "data/raw/esol.csv").exists() else "esol.csv"
    fasta_path = root / "data/raw/idmapping.fasta" if (root / "data/raw/idmapping.fasta").exists() else "idmapping.fasta"
    
    out = build_clean_dataset(raw_path, fasta_path)
    out.to_csv("data/processed/esol_cleaned.csv", index=False)
    print("Saved esol_cleaned.csv successfully! Shape:", out.shape)