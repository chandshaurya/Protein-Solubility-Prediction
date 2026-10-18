# Raw input data

This directory contains the source files used to connect the eSOL *E. coli*
K-12 solubility records with protein sequences:

- [`esol.csv`](./esol.csv) contains protein identifiers, measured solubility
  and yield values, protein annotations, and structural references.
- [`idmapping.fasta`](./idmapping.fasta) contains protein sequences in
  UniProt-style FASTA format. The preprocessing script uses the gene name in
  each FASTA header to match a sequence to an eSOL row.

The project uses `Solubility (%)` from `esol.csv` as its raw prediction target.
The other assay columns are additional measurements or condition-specific
values; they are not automatically used as targets by the current training
script.

## `esol.csv` columns

The CSV contains the following columns. Names and units below follow the
source file; condition abbreviations are retained as supplied by the dataset.
Blank cells indicate that a value is not supplied for that record. A blank is
not equivalent to a measured zero.

### Protein identifiers

| Column | Meaning |
| --- | --- |
| `JW_ID` | JW identifier for the gene/protein record in the eSOL dataset. |
| `ECK number` | ECK identifier for the *E. coli* K-12 record. |
| `B number` | B-number identifier for the *E. coli* K-12 gene. |
| `Gene name K-12` | Gene name used by the dataset; this is the field matched to the FASTA `GN=` tag by preprocessing. |
| `Locus name K-12` | K-12 locus name associated with the record. |
| `Synonyms of locus names K-12` | Alternate locus names supplied by the source. A cell may contain multiple synonyms. |

Identifiers and names are labels, not model measurements. Their formatting
should be preserved when joining tables; in particular, identifiers that look
numeric should not be treated as quantities.

### Solubility and yield measurements

| Column | Meaning |
| --- | --- |
| `Solubility (%)` | Main solubility percentage field used as the target in the included training workflow. |
| `Yield (uM)` | Protein yield in micromolar (`uM`) as reported in the source table. |
| `Yield (ug/ml)` | Protein yield in micrograms per milliliter (`ug/ml`) as reported in the source table. |
| `Minus Sol (%)` | Solubility percentage for the source's `Minus` assay condition. |
| `TF Sol (%)` | Solubility percentage for the `TF` assay condition. |
| `GroE Sol (%)` | Solubility percentage for the `GroE` assay condition. |
| `KJE Sol (%)` | Solubility percentage for the `KJE` assay condition. |
| `Minus Yield (uM)` | Yield in micromolar (`uM`) for the `Minus` assay condition. |
| `TF Yield (uM)` | Yield in micromolar (`uM`) for the `TF` assay condition. |
| `GroE Yield (uM)` | Yield in micromolar (`uM`) for the `GroE` assay condition. |
| `KJE Yield (uM)` | Yield in micromolar (`uM`) for the `KJE` assay condition. |
| `Minus Yield (ug/ml)` | Yield in micrograms per milliliter (`ug/ml`) for the `Minus` assay condition. |
| `TF Yield (ug/ml)` | Yield in micrograms per milliliter (`ug/ml`) for the `TF` assay condition. |
| `GroE Yield (ug/ml)` | Yield in micrograms per milliliter (`ug/ml`) for the `GroE` assay condition. |
| `KJE Yield (ug/ml)` | Yield in micrograms per milliliter (`ug/ml`) for the `KJE` assay condition. |

`TF`, `GroE`, `KJE`, and `Minus` are experimental condition labels from the
source dataset. When interpreting comparisons among these measurements, use
the assay definitions in the source study and dataset documentation. Do not
assume that a missing condition-specific value is zero or that it can be
substituted for `Solubility (%)`.

### Protein properties and annotations

| Column | Meaning |
| --- | --- |
| `Calculated MW (kDa)` | Calculated molecular weight in kilodaltons (`kDa`). |
| `Calculated pI` | Calculated isoelectric point (pI); it has no physical unit. |
| `Type of gene product` | Source categorical code describing the gene-product type. Retain as a category; use the source legend for the meanings of individual codes. |
| `Gene product description` | Text description of the gene product. |
| `Cell location` | Source annotation for the protein's cellular location. |
| `Structure (PDB) ID` | Protein Data Bank structure identifier, when supplied. |
| `SCOP assignment` | Structural Classification of Proteins (SCOP) assignment text and identifiers, when supplied. A cell may contain multiple assignments. |

These annotations provide context but are not all used by the current feature
extraction or model-training scripts.

## `idmapping.fasta` format and use

The FASTA file contains records with a header line beginning with `>` followed
by one or more amino-acid sequence lines. For example:

```text
>sp|P00547|KHSE_ECOLI Homoserine kinase OS=Escherichia coli (strain K12) OX=83333 GN=thrB PE=1 SV=2
MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGRFADKLPSEP
...
```

In a UniProt-style header:

- `sp` is the record source/type prefix.
- `P00547` is the accession.
- `KHSE_ECOLI` is the entry name.
- `OS=` gives the organism name, and `OX=` its taxonomy identifier.
- `GN=thrB` gives the gene name used by this project's join.
- `PE=` and `SV=` are UniProt protein-evidence and sequence-version fields.

The sequence itself can span multiple lines. The preprocessing parser joins
those lines into a single sequence for each record. It extracts the accession,
entry name, sequence, and gene tag, then matches the FASTA `GN=` gene name
against `Gene name K-12` in the CSV. A matching gene name is therefore required
for a sequence to be associated with an eSOL row.

The current parser keeps the first FASTA record found for a gene name. After
mapping, preprocessing retains sequences made only from the 20 standard amino
acids (`ACDEFGHIKLMNPQRSTVWY`); records with no match or an invalid sequence
are omitted from the sequence-merged output. This behavior can affect which
rows appear in the processed dataset.

## Rebuilding the processed tables

Run commands from the project root:

```powershell
python src/preprocessing.py
python src/features.py
```

`preprocessing.py` cleans the raw eSOL columns and maps sequences from this
FASTA file. `features.py` calculates sequence-derived features from the
processed table. See the root [README](../../README.md) for the complete
workflow, generated-file locations, expected target-column names, and model
usage.

The preprocessing step keeps a subset of `esol.csv` columns, renames several
fields, and writes derived tables under `data/processed/`; it does not preserve
every raw assay measurement listed above. Keep the raw files unchanged when
reproducing the project workflow.
