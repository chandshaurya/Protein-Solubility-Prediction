# Protein Solubility Prediction

A research and educational project for estimating the **in-vitro solubility of
*Escherichia coli* K-12 proteins** from their amino-acid sequences. It combines
the eSOL protein-solubility measurements with sequence-derived features, a
scikit-learn regression model, a Python inference API, and a Streamlit
application for interactive prediction and analysis.

> **Research use only.** Model outputs are estimates produced from a particular
> dataset and computational workflow. They are not laboratory measurements,
> clinical advice, or a substitute for validating an expression construct in
> the intended experimental system.

## Contents

- [What the project does](#what-the-project-does)
- [Project structure](#project-structure)
- [Data and provenance](#data-and-provenance)
- [Requirements and installation](#requirements-and-installation)
- [Quick start](#quick-start)
- [Rebuilding the data](#rebuilding-the-data)
- [Sequence features](#sequence-features)
- [Training and model artifact](#training-and-model-artifact)
- [Making predictions](#making-predictions)
- [Evaluation and interpretation](#evaluation-and-interpretation)
- [Tests](#tests)
- [Important workflow and scientific caveats](#important-workflow-and-scientific-caveats)
- [References](#references)

## What the project does

The central task is **regression**: given a protein sequence and its derived
features, predict a continuous solubility value on a 0–100 percentage scale.
The interface also presents a convenience label using a 50% threshold:

- **Soluble:** predicted solubility is greater than or equal to 50%.
- **Insoluble:** predicted solubility is below 50%.

That label is derived from the regression estimate; the application does not
train a separate classifier for the single-sequence prediction view.

The repository includes these workflows:

1. Clean eSOL records and map genes to sequences from a UniProt-style FASTA
   file.
2. Calculate sequence-composition, physicochemical, and positional features.
3. Train and save a scikit-learn histogram gradient boosting regressor.
4. Predict for a single sequence or batch of FASTA records.
5. Inspect available dataset summaries, evaluation metrics, plots, errors, and
   feature attributions in a Streamlit interface.

The full workflow, in broad terms, is:

```text
eSOL measurements ──┐
                    ├─ gene mapping and filtering ── sequence feature table
UniProt FASTA ──────┘                                      │
                                                           ├─ model training
                                                           ├─ model evaluation
                                                           └─ sequence inference
```

## Project structure

```text
.
├── app/
│   └── streamlit_app.py                 Interactive prediction and analytics UI
├── data/
│   ├── raw/
│   │   ├── esol.csv                     Source eSOL table
│   │   └── idmapping.fasta              Sequence records used for gene mapping
│   └── processed/
│       ├── desol_cleaned.csv            Cleaned eSOL rows before sequence merge
│       ├── esol_cleaned.csv              Cleaned rows with mapped sequences
│       ├── esol_with_sequences.csv       Included normalized sequence table
│       └── esol_advanced_features.csv    Included model-ready feature table
├── models/
│   └── best_model.joblib                Serialized estimator and feature names
├── results/
│   ├── figures/                          Saved evaluation and analysis plots
│   └── tables/                           Saved metrics and error-case tables
├── src/
│   ├── preprocessing.py                  eSOL cleaning and FASTA gene mapping
│   ├── features.py                       Sequence feature calculation
│   ├── train.py                          Baseline comparison and candidate training
│   ├── evaluate.py                       Holdout metrics and diagnostic plots
│   ├── depth_analysis.py                 Error, permutation, and SHAP analysis
│   └── predict.py                        Python inference API
├── tests/
│   └── test_experiment_design.py          Checks declared feature groups
└── requirements.txt                      Python package dependencies
```

Run the commands below from the project root unless a command explicitly says
otherwise. Several scripts use paths relative to the current working directory.

## Data and provenance

The project uses the eSOL dataset described by Niwa et al. (2009), paired with
protein sequences from the supplied FASTA mapping file. eSOL measurements
represent **chaperone-free protein expression in the PURE in-vitro system**;
they should not be interpreted as direct measurements of cellular expression
or solubility under arbitrary laboratory conditions.

The included processed feature table currently contains **2,642 records**.
Its target column is `Solubility (%)`. The table also contains identifiers and
descriptive metadata such as `gene`, `locus`, and `sequence`, in addition to
numeric feature columns.

### Source and processed tables

| File | Role |
| --- | --- |
| `data/raw/esol.csv` | Raw eSOL measurements and annotations. |
| `data/raw/idmapping.fasta` | UniProt-style FASTA records; the mapper reads the `GN=` gene tag. |
| `data/processed/desol_cleaned.csv` | Cleaned eSOL rows before sequences are mapped. |
| `data/processed/esol_cleaned.csv` | Cleaned eSOL rows with mapped sequence and accession fields when a FASTA file is available. |
| `data/processed/esol_with_sequences.csv` | Included normalized sequence table. Its target is named lowercase `solubility`. |
| `data/processed/esol_advanced_features.csv` | Included feature table used by the training and analysis scripts. It retains `Solubility (%)`. |

Preprocessing drops rows without a target, keeps the first row for duplicate
gene names, and retains sequences composed only of the 20 standard amino-acid
letters. The FASTA parser expects UniProt-style headers with a `GN=` gene tag.
When more than one FASTA record has the same gene name, the first parsed record
is retained.

## Requirements and installation

- Python 3.10 or newer
- Packages listed in `requirements.txt`
- The included data and model files for the ready-to-run prediction and analysis
  workflows

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script execution is restricted by the current PowerShell policy, activate
the environment using the policy approved for your machine, or invoke its
Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes the packages used by data processing, model
training, plotting and SHAP analysis, the Streamlit application, and tests.
Standard-library modules such as `pathlib`, `re`, and `logging` do not need to
be installed separately.

## Quick start

With dependencies installed and the included model and feature table present,
launch the application from the repository root:

```powershell
streamlit run app/streamlit_app.py
```

The app provides four views:

1. **Sequence Predictor** — enter one amino-acid sequence and view a predicted
   percentage, its threshold label, sequence length, and GRAVY score.
2. **Batch FASTA Predictor** — upload or paste FASTA records and download the
   returned predictions as CSV.
3. **Dataset Explorer** — inspect a preview and summary statistics of the
   processed feature table.
4. **Evaluation & Depth Analysis** — view saved metrics, diagnostic figures,
   SHAP/permutation feature importance, and high-error examples when those
   output files are present.

To run a prediction directly from Python, see
[Making predictions](#making-predictions). Training is not required when the
included compatible checkpoint is available.

## Rebuilding the data

The processed tables included with the project make it possible to use the
main training and inference workflows without starting from raw data. A
rebuild from source data is more involved because the preprocessing output
schema and the model-training schema are not fully automated into one command.

### 1. Clean the eSOL table and map sequences

From the project root:

```powershell
python src/preprocessing.py
```

The script reads `data/raw/esol.csv` and `data/raw/idmapping.fasta` when those
files are available. It writes the cleaned eSOL table to
`data/processed/desol_cleaned.csv` and, when the FASTA is available, the
sequence-merged table to `data/processed/esol_cleaned.csv`.

The normalized output uses `solubility` (lowercase) for its target. The
training and evaluation scripts expect `Solubility (%)`. Therefore, the
preprocessing output is **not a direct replacement** for the included
model-ready feature table; if rebuilding from raw data, make the expected target
and feature schema explicit before running training.

### 2. Calculate sequence features

The feature-generation helper defaults to reading
`data/processed/esol_cleaned.csv` and writing
`data/processed/esol_advanced_features.csv`:

```powershell
python src/features.py
```

The input must have a `sequence` or `Sequence` column. The generated feature
table preserves the columns from the input and appends calculated features.
Feature extraction alone does not rename the normalized `solubility` target to
the `Solubility (%)` field expected by the training scripts.

For custom input and output paths, call the helper from Python:

```python
from src.features import generate_full_feature_dataset

features = generate_full_feature_dataset(
    input_csv="data/processed/esol_cleaned.csv",
    output_csv="data/processed/esol_advanced_features.csv",
)
print(features.shape)
```

When rebuilding, verify the output columns and target values before training.
Avoid replacing the included model-ready table until the rebuilt schema has
been checked.

## Sequence features

`src/features.py` exposes
`extract_advanced_physicochemical_features(sequence)`. It uppercases the
sequence and filters out symbols outside the standard 20 amino acids. Sequences
with fewer than five retained amino acids do not produce a feature dictionary.

For a sufficiently long sequence, the extractor calculates up to 497 derived
features:

| Feature family | Contents |
| --- | --- |
| Length | `seq_length` and `log_seq_length`. |
| Amino-acid composition (AAC) | Fraction of each of the 20 standard amino acids (`aac_A`, `aac_C`, and so on). |
| Dipeptide composition | Fraction of each possible adjacent pair: 400 `di_XX` features. |
| Physicochemical summaries | Kyte–Doolittle-based `gravy_score`, approximate `net_charge_density`, and `hydrophobic_std`. |
| Autocorrelation | Hydropathy and charge products over residue separations from 1 through at most 15. |
| CTD groups | Composition for seven residue groups and five positional quantiles per group. |

The model-ready table also includes reported properties such as molecular
weight (`mw_kda_reported`) and isoelectric point (`pi_reported`). Missing
numeric values are median-filled by the training and analysis workflows.

The extractor’s filtering behavior is useful for permissive input handling,
but it is not strict validation: invalid symbols are discarded rather than
causing the whole sequence to be rejected. A submitted sequence containing
valid and invalid characters can therefore be treated as the shorter sequence
made from its valid residues.

## Training and model artifact

Train the candidate model from the project root:

```powershell
python src/train.py
```

The script loads `esol_advanced_features.csv` from the project root if present;
otherwise it reads `data/processed/esol_advanced_features.csv`. It selects
numeric columns other than metadata and the target, median-fills missing
feature values, and uses `Solubility (%)` as the raw target.

### Training procedure

The current script:

1. Smooths each target using the mean target among its 12 nearest neighbors in
   feature space (`NearestNeighbors`).
2. Splits the resulting rows randomly into 85% training and 15% test data using
   `random_state=42`.
3. Fits a Ridge regression baseline (`alpha=1.0`) and a
   `HistGradientBoostingRegressor` candidate.
4. Reports R², RMSE, and MAE for the holdout predictions.
5. Saves the candidate estimator and ordered feature names in
   `models/best_model.joblib`.

Candidate model settings currently include `max_iter=1200`,
`learning_rate=0.01`, `max_depth=8`, `min_samples_leaf=20`,
`l2_regularization=1.0`, and `random_state=42`.

The checkpoint is a joblib-serialized dictionary with two keys:

- `model`: the fitted scikit-learn estimator.
- `feature_names`: ordered training feature names required at inference time.

`train.py` overwrites `models/best_model.joblib` when it runs. Keep a copy of
any checkpoint that must be preserved before retraining. The files under
`models/` and `results/` can reflect different experiment runs, so a metric
file should not be assumed to describe whichever checkpoint currently happens
to be loaded.

## Making predictions

### Single sequence from Python

```python
from src.predict import SolubilityPredictor

predictor = SolubilityPredictor("models/best_model.joblib")
prediction = predictor.predict_from_sequence(
    "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLPSEP"
)

print(prediction)
```

The returned dictionary includes:

- `predicted_solubility_percent` — prediction clipped to the 0–100 range and
  rounded to two decimal places.
- `is_soluble` — `True` when the clipped estimate is at least 50%.
- `seq_length` — number of standard amino-acid residues used for features.
- `gravy_score` — mean hydropathy summary.
- `net_charge_density` — the extractor’s approximate mean charge score.

The model path may be supplied explicitly. Relative model paths are searched
from the project root, the current working directory, and the `src` directory.
The default checkpoint path is `models/best_model.joblib`.

### Batch FASTA from Python

```python
from pathlib import Path
from src.predict import SolubilityPredictor

predictor = SolubilityPredictor()
fasta_text = Path("proteins.fasta").read_text(encoding="utf-8")
predictions = predictor.predict_batch_fasta(fasta_text)
predictions.to_csv("protein_solubility_predictions.csv", index=False)
```

Each FASTA entry is represented by the first word in its header. The resulting
DataFrame includes the protein ID, predicted percentage, threshold
classification, sequence length, GRAVY score, charge density, and original
sequence. The current batch helper skips records that raise an exception
(including sequences too short to generate features); check the number of
returned rows against the number of input records if every record must be
accounted for. Batch prediction is an inference helper, not a strict FASTA
validator.

The app accepts FASTA files with `.fasta`, `.fa`, or `.txt` extensions, or
pasted FASTA text. It offers a CSV download of the results.

## Evaluation and interpretation

### Holdout evaluation

Run:

```powershell
python src/evaluate.py
```

The script loads `models/best_model.joblib` and
`data/processed/esol_advanced_features.csv`. It recreates the 85/15
`random_state=42` split, trains a Ridge baseline on the training partition,
and compares the loaded candidate model with that baseline.

It writes:

| Output | Description |
| --- | --- |
| `results/tables/comparative_metrics.csv` | Ridge and candidate regression and threshold-classification metrics. |
| `results/figures/comparative_parity_plot.png` | Baseline and candidate predictions against the target. |
| `results/figures/parity_plot.png` | Candidate-only parity plot. |
| `results/figures/residuals_distribution.png` | Distribution of candidate residuals. |
| `results/figures/roc_auc_curve.png` | ROC curves for threshold labels, using regression predictions as scores. |
| `results/figures/confusion_matrix.png` | Candidate classification counts using the 50% threshold. |

Regression metrics include R², RMSE, and MAE. Classification metrics use
`Solubility (%) >= 50` as the positive label. These derived binary metrics do
not mean the model was trained as a classifier.

The app can also display `results/tables/evaluation_metrics.csv` when the
comparative metrics file is absent. That file is a separate saved artifact; the
current `src/evaluate.py` writes `comparative_metrics.csv`, not
`evaluation_metrics.csv`.

### Feature and error analysis

Run:

```powershell
python src/depth_analysis.py
```

This script is rooted at the project directory when resolving its input and
output paths and performs the following analysis:

- Saves `results/tables/top_10_high_error_cases.csv`.
- Calculates permutation feature importance and saves the top-feature plot to
  `results/figures/feature_importance.png`.
- Uses SHAP’s permutation explainer to calculate global attributions for a
  deterministic sample of up to 25 test records, with up to 20 training
  records as the background. It saves a beeswarm plot to
  `results/figures/shap_summary_plot.png`.
- Saves `results/figures/gravy_vs_solubility.png` when `gravy_score` is
  present.

SHAP is model-agnostic here because the current candidate is a histogram
gradient boosting estimator. With hundreds of input features, permutation
explanations require many model evaluations; this analysis can take longer
than the training or ordinary evaluation steps. The deterministic sample
keeps the work bounded and reproducible, but the plot is an approximation over
that selected subset.

Permutation importance and SHAP describe how the fitted model behaves on the
chosen data. They do not establish that a feature causes solubility changes or
that an attribution corresponds to a biological mechanism.

### Saved result snapshots

The checked-in tables are outputs from particular saved runs. For example, the
current `results/tables/comparative_metrics.csv` contains a Ridge baseline
R² of 0.645 and a candidate R² of 0.763; the separate
`results/tables/evaluation_metrics.csv` contains a holdout R² of 0.766.
These snapshots may correspond to different checkpoints, code versions, or
evaluation protocols. Re-running a script can replace its outputs, and the
numbers are not guaranteed to reproduce across differing data or model
artifacts.

## Tests

Run the available test suite from the project root:

```powershell
python -m pytest -q
```

The current test verifies that the declared feature experiment groups in
`src/depth_analysis.py` contain the expected feature categories. It is a
focused structural test, not a full end-to-end test of data rebuilding,
training, evaluation, or the Streamlit interface.

## Important workflow and scientific caveats

### Target smoothing and holdout leakage

Both training and evaluation scripts smooth raw targets with nearest-neighbor
information **before** making their random train/test split. A row’s smoothed
target can therefore be influenced by neighboring rows that later appear in
the test partition. This means the holdout does not represent a fully
independent evaluation of the original measurements and can yield optimistic
performance estimates. Treat scores from this workflow as exploratory, not as
unbiased generalization estimates.

### Split and related proteins

The split is a random row split, not a protein-family-aware or sequence
similarity-aware split. Related proteins may appear in both training and test
partitions. Evaluating generalization to novel families would require a
similarity-aware split and a leakage-safe target-construction procedure.

### Data and artifact alignment

The training, evaluation, and inference scripts rely on consistent feature
columns and the ordered feature names saved with the model. If the input data
or feature-generation implementation changes, validate that the model,
feature table, and evaluation outputs belong to the same run before
interpreting results.

The current data rebuild process has a schema boundary: preprocessing emits a
lowercase `solubility` target, while the existing training scripts expect
`Solubility (%)`. Rebuilding feature data does not automatically reconcile
that naming difference or guarantee equivalence with the included feature
table.

### Interpretation boundaries

Solubility and expression depend on factors not represented by sequence alone,
including assay conditions, construct design, cellular environment, folding
kinetics, and expression system. The eSOL measurements concern one specific
chaperone-free in-vitro system. Predictions should not be generalized
automatically to cellular expression, other organisms, or different assays.

## References

- Niwa, T. et al. (2009). “Global analysis of the effects of the molecular
  chaperone machines on the folding of the *Escherichia coli* proteome.”
  *Proceedings of the National Academy of Sciences*.
  [https://doi.org/10.1073/pnas.0811922106](https://doi.org/10.1073/pnas.0811922106)
- UniProt Consortium. UniProt Knowledgebase.
  [https://www.uniprot.org/](https://www.uniprot.org/)
