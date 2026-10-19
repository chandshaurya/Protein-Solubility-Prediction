<p align="center">
  <img src="assets/protein-solubility-logo.svg" alt="Protein Solubility Prediction logo" width="760">
</p>

<h1 align="center">Protein Solubility Prediction</h1>
<p align="center"><strong>Sequence-based regression, model interpretation, and an interactive prediction app</strong></p>
<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
<img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
<img alt="Research" src="https://img.shields.io/badge/Use-research%20%26%20education-64748b">
</p>

A research and educational project that estimates **in-vitro solubility of *Escherichia coli* K-12 proteins** from amino-acid sequences. It combines eSOL measurements, sequence-derived features, scikit-learn regression, evaluation and interpretation scripts, a Python inference API, and a Streamlit interface.

> **Research use only:** predictions are computational estimates, not laboratory measurements. eSOL measurements were obtained in a particular chaperone-free cell-free PURE system; results should not be assumed to predict solubility in every organism or experimental setup.

## Contents
- [Project overview](#project-overview)
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Dataset and provenance](#dataset-and-provenance)
- [Feature engineering](#feature-engineering)
- [Models and evaluation](#models-and-evaluation)
- [Installation and quick start](#installation-and-quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Prediction examples](#prediction-examples)
- [App features](#app-features)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Troubleshooting](#troubleshooting)
- [Scientific caveats](#scientific-caveats)
- [Reproducibility checklist](#reproducibility-checklist)
- [Citation and attribution](#citation-and-attribution)
- [License](#license)

## Project overview

**Task:** regression. Given a protein's amino-acid sequence and derived numerical features, predict a continuous solubility value. The UI also derives a convenient label from the estimate: `Soluble` when the predicted value is at least 50%, otherwise `Insoluble`. This is a post-prediction threshold; the single-sequence interface does not train a separate classifier.

| Component | Current implementation |
|---|---|
| Source measurements | eSOL, provided by Japan's Life Science Database Archive (LSDB Archive) |
| Included processed table | 2,642 records |
| Target expected by training scripts | `Solubility (%)` |
| Baseline | Ridge regression (`alpha=1.0`) |
| Candidate | `HistGradientBoostingRegressor` |
| Feature groups | Length, AAC, dipeptides, physicochemical summaries, autocorrelations, CTD-style descriptors, reported properties |
| Interpretation | Permutation importance, SHAP approximation, ablation, residual and error analysis |
| Interfaces | Python prediction API and Streamlit app |

The 2,642 records refer to the **included processed table**, not the total size of the original eSOL archive.

## Pipeline

```text
eSOL measurements ─────┐
                       ├─ cleaning + gene-to-sequence mapping
UniProt-style FASTA ───┘
                              |
                              v
                      normalized sequences
                              |
                              v
                      numerical features
                              |
                 +------------+-------------+
                 v            v             v
              training    evaluation   depth analysis
                 |            |             |
                 v            v             v
             model file   metrics/plots  SHAP/ablation
                 |
                 v
        Python API + Streamlit app
```

## Repository structure

```text
.
├── app/streamlit_app.py
├── assets/protein-solubility-logo.svg
├── data/
│   ├── raw/
│   │   ├── esol.csv
│   │   └── idmapping.fasta
│   └── processed/
│       ├── desol_cleaned.csv
│       ├── esol_cleaned.csv
│       ├── esol_with_sequences.csv
│       └── esol_advanced_features.csv
├── models/best_model.joblib
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── depth_analysis.py
│   └── predict.py
├── tests/test_experiment_design.py
├── requirements.txt
└── README.md
```

Some directories and output files may only appear after running scripts. Execute commands from the repository root because several scripts use paths relative to the current working directory.

## Dataset and provenance

The project uses the [official eSOL solubility dataset](https://dbarchive.biosciencedbc.jp/en/esol/data-1.html) from the [LSDB Archive](https://dbarchive.biosciencedbc.jp/en/esol/download.html). eSOL describes solubility measurements from a chaperone-free, reconstituted cell-free translation system (PURE system). These measurements are assay-specific.

| File | Role |
|---|---|
| `data/raw/esol.csv` | Raw eSOL measurements and annotations |
| `data/raw/idmapping.fasta` | UniProt-style FASTA used for gene mapping |
| `data/processed/desol_cleaned.csv` | Cleaned eSOL rows before sequence merge |
| `data/processed/esol_cleaned.csv` | Cleaned rows with mapped sequences when available |
| `data/processed/esol_with_sequences.csv` | Normalized sequence table; target named `solubility` |
| `data/processed/esol_advanced_features.csv` | Model-ready table; target expected as `Solubility (%)` |

Preprocessing drops rows without a target, keeps the first row for duplicate gene names, and retains sequences composed only of the 20 standard amino-acid letters. The FASTA parser expects a `GN=` gene tag and keeps the first record when duplicate gene names occur.

### Obtaining the data
1. Open the [official eSOL download page](https://dbarchive.biosciencedbc.jp/en/esol/download.html).
2. Read the [official license](https://dbarchive.biosciencedbc.jp/en/esol/lic.html).
3. Download the eSOL table and obtain the sequence FASTA used for this project.
4. Place them at `data/raw/esol.csv` and `data/raw/idmapping.fasta`, or update the paths in the preprocessing script.

**Attribution:** the eSOL archive has its own attribution and share-alike requirements. Follow the current official license terms when redistributing data or derived materials, and credit Hideki Taguchi as specified by the archive.

## Feature engineering

`src/features.py` exposes `extract_advanced_physicochemical_features(sequence)`. It uppercases the sequence and filters characters outside the standard 20 amino acids. Fewer than five retained residues do not produce a feature dictionary.

The extractor calculates up to 497 derived features:

| Family | Features |
|---|---|
| Length | `seq_length`, `log_seq_length` |
| Amino-acid composition (AAC) | 20 amino-acid fractions, e.g. `aac_A` |
| Dipeptide composition | 400 adjacent-pair frequencies, `di_XX` |
| Physicochemical summaries | Kyte–Doolittle `gravy_score`, approximate `net_charge_density`, `hydrophobic_std` |
| Autocorrelation | Hydropathy and charge products across residue separations up to 15 |
| CTD-style descriptors | Composition of seven residue groups across five sequence quantiles |

The model-ready table can also contain reported properties such as `mw_kda_reported` and `pi_reported`. Numeric missing values are median-filled in current training and analysis workflows.

**Input caveat:** invalid symbols are filtered instead of strictly rejecting the entire sequence. A sequence with mixed valid and invalid characters may silently become a shorter sequence; validate sequences before scientific use.

## Models and evaluation

Current scripts compare Ridge regression (`alpha=1.0`) with a histogram gradient boosting regressor configured as follows:

| Parameter | Value |
|---|---:|
| `max_iter` | 1200 |
| `learning_rate` | 0.01 |
| `max_depth` | 8 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 1.0 |
| `random_state` | 42 |
| Split | 85% train / 15% test |
| Target smoothing | 12 nearest neighbors |

The current main workflow smooths targets using nearest neighbors **before** the random split. That creates a potential information leak, so the resulting holdout metrics must be treated as exploratory rather than as an unbiased estimate of generalization. Details are in [Scientific caveats](#scientific-caveats).

`models/best_model.joblib` contains a dictionary with `model` (the fitted estimator) and `feature_names` (the ordered input feature names). `src/train.py` overwrites this checkpoint when rerun; back it up first if needed.

## Installation and quick start

Python 3.10+ is recommended.

### Windows PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked by policy, use the environment's Python executable directly:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the app from the repository root:
```bash
streamlit run app/streamlit_app.py
```
The app uses saved model/data/results; launching it does not automatically retrain the model.

## Reproducing the experiments

Follow the steps in order. Run commands from the **repository root** (the folder containing `src/`, `app/`, and `requirements.txt`). The commands below use the included project structure. The current repository does **not** rebuild every artifact with one end-to-end command; data preparation, feature generation, training, evaluation, and deeper analysis are separate stages.

### Step 1 — Create and activate a virtual environment

A virtual environment keeps this project's Python packages isolated from other projects.

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python executable directly in the later commands:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

With the virtual environment activated, run:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that Python is using the environment you created:
```bash
python --version
python -m pip --version
```

Use Python 3.10 or newer. If activation was blocked on Windows, replace `python` with `.\.venv\Scripts\python.exe` in the commands below.

### Step 3 — Check the input files

Confirm that the expected files are present:
```text
data/raw/esol.csv
data/raw/idmapping.fasta
```

The repository may already include processed data, a trained model, and saved result files. If you only want to launch the existing app, skip to **Step 8 — Launch Streamlit**. To rebuild the experiment from raw data, continue below.

### Step 4 — Clean data and map sequences

```bash
python src/preprocessing.py
```

Reads `data/raw/esol.csv` and `data/raw/idmapping.fasta` when available. Writes `data/processed/desol_cleaned.csv` and, when the FASTA is available, `data/processed/esol_cleaned.csv`.

**Schema checkpoint:** this preprocessing stage outputs lowercase `solubility`, while training expects `Solubility (%)`. Explicitly reconcile and verify the schema before proceeding; do not blindly replace the included model-ready table.

### Step 5 — Generate sequence features

```bash
python src/features.py
```

Requires a `sequence` or `Sequence` column and writes `data/processed/esol_advanced_features.csv` by default. It appends features but does not automatically rename `solubility` to `Solubility (%)`.

Before training, check target naming, row alignment, numeric predictors, missing values, and metadata exclusion.

### Step 6 — Train the model

```bash
python src/train.py
```

The current script selects numeric predictors, median-fills missing features, smooths the target with the mean target of 12 nearest neighbors in feature space, splits rows 85/15 with seed 42, trains Ridge and HistGradientBoosting regressors, reports R²/RMSE/MAE, and saves `models/best_model.joblib`.

**Important:** the current workflow smooths targets before the split, which creates a potential information leak. Treat its holdout scores as exploratory rather than unbiased estimates of generalization.

### Step 7 — Evaluate and run feature/error analysis

Evaluate the saved checkpoint:
```bash
python src/evaluate.py
```

This loads the model and feature table, recreates the seeded split, fits the Ridge baseline, compares predictions, and writes metrics and plots. Because target smoothing occurs before splitting, this is not a leakage-safe evaluation.

Common evaluation outputs:
- `results/tables/comparative_metrics.csv`
- `results/figures/comparative_parity_plot.png`
- `results/figures/parity_plot.png`
- `results/figures/residuals_distribution.png`
- `results/figures/roc_auc_curve.png`
- `results/figures/confusion_matrix.png`

Run deeper feature/error analysis:
```bash
python src/depth_analysis.py
```

This generates high-error cases, feature-family ablation, bootstrap RMSE intervals, permutation importance, a SHAP summary, and a GRAVY-versus-solubility plot when the required inputs and dependencies are available. The ablation uses measured targets on a fixed holdout rather than the smoothed targets used by the main workflow; its metrics represent a separate protocol and should not be compared as if they were the same experiment.

### Step 8 — Launch the Streamlit application

From the repository root:
```bash
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL in the terminal (usually `http://localhost:8501`). Open that URL in your browser.

The app provides these views when their supporting files are available:
1. **Sequence Predictor** — enter a protein sequence and view the estimated solubility and derived threshold label.
2. **Batch FASTA Predictor** — submit FASTA records and export predictions as CSV.
3. **Dataset Explorer** — inspect the processed dataset and summary statistics.
4. **Evaluation & Depth Analysis** — explore saved metrics, plots, feature attributions, and high-error examples.

The app uses the saved model, processed data, and result files; launching it does not automatically retrain the model. If you only want to use the existing application and its saved checkpoint, Steps 1–2 and Step 8 are normally sufficient, provided the required model/data files are present.

### Reproduction checklist

- [ ] Create and activate the virtual environment.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Confirm the raw input files or included processed tables are present.
- [ ] Verify the target column name after preprocessing.
- [ ] Back up the current model and results before retraining.
- [ ] Run each script from the repository root.
- [ ] Confirm metrics and plots correspond to the same code, data, model, and run.
- [ ] Document the target-smoothing-before-split limitation when reporting results.
- [ ] Launch Streamlit and verify the required model and artifacts load.

## Prediction examples

### One sequence
```python
from src.predict import SolubilityPredictor

predictor = SolubilityPredictor("models/best_model.joblib")
result = predictor.predict_from_sequence(
    "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLPSEP"
)
print(result)
```

The returned dictionary includes `predicted_solubility_percent` (clipped to 0–100 and rounded), `is_soluble` (threshold 50%), `seq_length`, `gravy_score`, and `net_charge_density`.

### Batch FASTA
```python
from pathlib import Path
from src.predict import SolubilityPredictor

predictor = SolubilityPredictor()
fasta_text = Path("proteins.fasta").read_text(encoding="utf-8")
predictions = predictor.predict_batch_fasta(fasta_text)
predictions.to_csv("protein_solubility_predictions.csv", index=False)
```
Batch parsing uses the first word of each FASTA header as the record ID. Some records may be skipped if feature generation raises an exception; compare input and output record counts.

## App features

Run `streamlit run app/streamlit_app.py`. The app provides these views when supporting artifacts are available:

1. **Sequence Predictor:** estimate, threshold label, sequence length, and GRAVY.
2. **Batch FASTA Predictor:** upload/paste FASTA and export predictions as CSV.
3. **Dataset Explorer:** preview the processed table and summary statistics.
4. **Evaluation & Depth Analysis:** saved metrics, diagnostic plots, feature attribution, and high-error examples.

## Outputs and interpretation

- **R²:** variance explained relative to a mean baseline on the evaluated partition.
- **RMSE:** root mean squared error; larger errors receive more weight.
- **MAE:** mean absolute prediction error.
- **Threshold metrics:** accuracy, F1, ROC-AUC, and confusion matrix are derived using the 50% threshold on regression predictions; the single-sequence model is not a separately trained classifier.

Permutation importance, SHAP, and ablation describe model behavior on selected data. They do not prove causation or identify a biological mechanism. The SHAP permutation explainer uses a bounded sample and is an approximation.

Existing snapshots report R² of 0.645 for Ridge and 0.763 for the candidate in `comparative_metrics.csv`, while a separate evaluation file reports 0.766. These files may come from different runs/checkpoints or protocols; do not treat them as one guaranteed result until the artifacts are aligned and rerun.

## Troubleshooting

| Problem | Check |
|---|---|
| `FileNotFoundError` | Run from repository root and confirm the expected file exists. |
| Missing target column | Preprocessing emits `solubility`; training expects `Solubility (%)`. |
| Feature mismatch at prediction | Compare input columns with the ordered `feature_names` saved in the checkpoint. |
| App plots missing | Confirm the corresponding `results/figures/` or `results/tables/` files exist. |
| Fewer batch predictions than input records | Check for invalid or too-short sequences that the batch helper skipped. |
| SHAP takes a long time | Permutation explainers call the model repeatedly; this stage may be slower than training. |

## Scientific caveats

1. **Target leakage risk:** target smoothing occurs before the random train/test split. A training target can be influenced by a row later placed in the test partition, potentially making metrics optimistic.
2. **Random split:** the split is not protein-family-aware or sequence-similarity-aware. Related proteins may occur in both partitions.
3. **Data rebuild schema:** preprocessing uses `solubility`, while training expects `Solubility (%)`; the full raw-to-model pipeline is not currently one-command reproducible.
4. **Artifact alignment:** model, feature table, metrics, and plots may originate from different runs.
5. **Assay scope:** eSOL is an in-vitro PURE-system measurement. Results should not automatically be generalized to cellular expression, other organisms, or different assay conditions.

## Roadmap for a stronger research release

- [ ] Split first; fit any target transformation only using training data.
- [ ] Add sequence-similarity-aware or protein-family-aware validation.
- [ ] Automate preprocessing → schema validation → feature generation → training → evaluation.
- [ ] Save a run manifest with data hashes, package versions, parameters, seed, and checkpoint ID.
- [ ] Add end-to-end tests on a small fixture dataset.
- [ ] Validate sequences strictly and report skipped FASTA entries.
- [ ] Regenerate all headline metrics and figures from one documented run.

## Citation and attribution

**Primary paper**

Niwa, T., Ying, B.-W., Saito, K., Jin, W., Takada, S., Ueda, T., and Taguchi, H. (2009). “Bimodal protein solubility distribution revealed by an aggregation analysis of the entire ensemble of *Escherichia coli* proteins.” *Proceedings of the National Academy of Sciences*, 106(11), 4201–4206. https://doi.org/10.1073/pnas.0811922106

**Official dataset:** [eSOL at LSDB Archive](https://dbarchive.biosciencedbc.jp/en/esol/data-1.html)  
**Data license:** [Official eSOL license page](https://dbarchive.biosciencedbc.jp/en/esol/lic.html)

When redistributing the dataset or derivative data, follow the archive's current attribution and share-alike requirements. Cite the data source separately from this software project.

## License

This project’s source code is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the terms and conditions of the MIT License.
