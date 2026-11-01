# Team Information
**Team Number:** Team 4

**Team Members:**
- Timothy Chan
- John Brisebois
- Shivam Shelar


# Setup
Tested with Python 3.14.8.

```bash
git clone https://github.com/TimothyChan2912/CS271-Team4-ClassifyingSentiment.git
cd CS271-Team4-ClassifyingSentiment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

# Data Access
1. Download the Dataset
- https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
- Click Download, then "Download dataset as zip", or use the Kaggle CLI (needs `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d kashishparmar02/social-media-sentiments-analysis-dataset -p data/raw --unzip
```

2. Extract the Dataset
- Unzip so the file is at `data/raw/sentimentdataset.csv`
- `src/data_cleaning.py` checks its SHA-256 (`725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b`) and stops if it differs, since Kaggle files can be updated in place.

3. Dataset Organization
The project uses the following directory structure:

```text
data/
├── raw/          # Original dataset downloaded from Kaggle
└── processed/    # Cleaned and transformed datasets
```

4. Dataset License and Attribution
- Source: Kaggle
- Dataset: Social Media Sentiments Analysis Dataset
- Author: Kashish Parmar
- License: CC0 1.0 Universal (Public Domain Dedication)
- Dataset URL: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset

5. Data Processing
The original dataset should remain unchanged in data/raw/. Any cleaning or processing done via the Python script should be saved separately in data/processed/.

# Baseline Run Routes
Run everything from the repository root.

**One command:** `python run_all.py` (options: `--remake-split` regenerates the split first; `--skip-b2` skips the TweetEval model download)

| Step | Command | Output |
|---|---|---|
| Clean data | `python src/data_cleaning.py` | `data/processed/clean.parquet`, `evidence/prepare_data_counts.json` |
| Make split (only if regenerating) | `python src/make_split.py` | `splits/split.csv`, `evidence/split_summary.json` |
| Check split | `python src/check_split.py > evidence/split_check_output.txt` | `All split checks passed.` |
| B1: logistic regression | `python src/baseline_logreg.py` | `evidence/b1_*` |
| B2: TweetEval RoBERTa, zero-shot | `python src/baseline_tweeteval.py` | `evidence/b2_tweeteval_metrics.json`, `evidence/b2_tweeteval_predictions.csv` |

All baselines score the dev partition only, load data through `src/data_io.py` (test rows are removed before any text is read), and compute metrics through `src/metrics.py`. The protocol is in `PROTOCOL.md`; expected values for every evidence file are in `evidence/EVIDENCE_RECORD.md`.


## B1 implementation and reproduction

The B1 entry point listed above, `src/baseline_logreg.py`, runs the scikit-learn
TF-IDF/logistic-regression baseline. It uses the existing `load_dev` and
`cv_report` functions. `python run_all.py --skip-b2` runs data preparation,
split checks, and B1. B1 was also verified with Python 3.13.7 on Windows.

Run the focused checks with `python -m unittest discover -s tests -v`.
The fixed settings are in `configs/baseline_logreg.json`.
A fresh pipeline is fitted on four dev folds and scored on the fifth for each
of the five saved folds. A convergence warning stops the run.

B1 writes directly to `evidence/`, producing `b1_logreg_predictions.csv`
(521 held-out dev predictions) and `b1_logreg_metrics.json` (shared
pooled/per-fold metrics, evaluation count, configuration, data/split identifiers,
runtime, CPU information, and output paths). Each run replaces these two files,
matching B2's output layout. Reports use B2's `pooled` and `per_fold_macro_f1*` fields.

Loader clarification: `load_dev` reads the cleaned Parquet file internally,
then returns the dev rows. Only those returned rows enter B1 training and scoring.
