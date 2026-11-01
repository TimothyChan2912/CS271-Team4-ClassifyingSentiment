# Evidence Record: Data Cleaning

## 1. Field Definitions

| Key | Meaning |
|---|---|
| `raw` | Rows in the downloaded CSV before any cleaning. |
| `after_empty_removed` | Rows remaining after dropping null or empty text. |
| `conflicting_groups` | Groups of near-identical texts with different emotion labels (all copies are dropped). |
| `after_dedup` | Rows remaining after removing exact and normalized duplicates (lowercased, URLs and @mentions stripped). |
| `dropped_ambiguous` | Rows removed because their emotion was mapped to `drop` in `configs/emotion_map.csv`. |
| `final` | Rows in `data/processed/clean.parquet`, which is the input to the splitting step. |
| `class_counts` | Number of rows in each sentiment class in the final dataset. |
| `posts_with_emoji` | Final posts containing at least one emoji. |
| `raw_sha256` | SHA-256 hash of the original raw CSV, used to verify that everyone is working with the same dataset file. |

## 2. Reproducibility Information

- **Produced by:** `python src/data_cleaning.py`.
- **Inputs:** `data/raw/sentimentdataset.csv` and `configs/emotion_map.csv`.
- **Output:** `evidence/prepare_data_counts.json`.
- **Expected values:** The values recorded in `prepare_data_counts.json` should match exactly when anyone reruns the cleaning script with the same inputs and configuration.

## 3. Important Notes

The cleaning results depend on the label mapping in `configs/emotion_map.csv`.

If this file is modified, the following values may change:

- `dropped_ambiguous`
- `final`
- `class_counts`

Any changes to the cleaning configuration or upstream data should be followed by rerunning the cleaning script and regenerating the evidence record. The data split should also be regenerated if the final cleaned dataset changes.

The `raw_sha256` field identifies the original raw dataset file. If the hash differs between runs, check whether the raw dataset file or its version has changed.

## 4. Split Summary (`evidence/split_summary.json`)

| Key | Meaning |
|---|---|
| `partition_sizes` | Number of posts in the `dev` and `test` partitions. |
| `class_counts_by_partition` | Posts per sentiment class in each partition. |
| `class_counts_by_fold` | Posts per sentiment class in each of the 5 dev folds (the held-out fold in each CV round). |
| `settings` | Seed, number of folds, and test fraction used to create the split. |

- **Produced by:** `python src/make_split.py`.
- **Inputs:** `data/processed/clean.parquet`.
- **Outputs:** `splits/split.csv` and `evidence/split_summary.json`.
- **Expected values:** dev 521 posts (negative 145, neutral 32, positive 344), test 131 posts (negative 36, neutral 8, positive 87); seed 42, 5 folds, test fraction 0.2. A rerun must produce an identical `split.csv`.

## 5. Split Boundary Check (`evidence/split_check_output.txt`)

- **Produced by:** `python src/check_split.py > evidence/split_check_output.txt`.
- **What it checks:** split IDs match the cleaned data and contain no duplicates; no post is in both dev and test; no identical or near-duplicate (normalized) text appears in both dev and test; all 5 folds exist; every fold contains all three classes; test posts all have fold -1; rebuilding the split reproduces the committed `split.csv` exactly.
- **Expected output:** `All split checks passed.` If any check fails, the script stops with an assertion message naming the problem, and the split must not be used.

## 6. Baseline B2: TweetEval RoBERTa, zero-shot (`evidence/b2_tweeteval_metrics.json`)

- **Produced by:** `python src/baseline_tweeteval.py`.
- **Inputs:** `data/processed/clean.parquet`, `splits/split.csv` (dev rows only; test rows are filtered out before text is loaded).
- **Outputs:** `evidence/b2_tweeteval_predictions.csv` (one row per dev post with class probabilities) and `evidence/b2_tweeteval_metrics.json`.
- **Contents:** pooled macro-F1, accuracy, per-class F1, and confusion matrix; per-fold macro-F1 with mean and sample std; positive-vs-negative accuracy (context for B0 only, not comparable); model commit, device, and library versions.
- **Expected values:** deterministic for a fixed model commit. CPU vs. GPU floating-point differences can flip a few borderline predictions; the device is recorded.

## 7. Teammate Rerun

Fresh clone, new virtual environment, `python run_all.py --remake-split --skip-b2`, then `git diff --exit-code splits/ evidence/prepare_data_counts.json evidence/split_summary.json`.

| Check | Expected | Observed |
|---|---|---|
| Raw SHA-256 | `725c6f7b...5275c63b` | |
| `final` rows | 652 | |
| Class counts | positive 431, negative 181, neutral 40 | |
| `split.csv` identical to committed file | yes (no diff output) | |
| `check_split.py` | `All split checks passed.` | |

Rerun by: ______ on ______ (OS, Python version: ______).

## 8. B1: baseline_logreg (2026-11-01)

`baseline_logreg` uses scikit-learn TF-IDF and logistic regression with the
saved development folds and shared evaluation metrics.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_all.py --skip-b2
```

The commands above reproduce B1 and its checks. Each run writes
`evidence/b1_logreg_predictions.csv` and `evidence/b1_logreg_metrics.json`,
replacing the previous contents.

| Field | Observed value |
|---|---|
| Baseline / run ID | baseline_logreg / baseline_logreg_20261101T231438867884Z |
| Data version | kaggle-social-2026-10-30 |
| Configuration | `configs/baseline_logreg.json`; one fixed configuration, no sweep |
| Split seed / model randomness | 42 / none (L-BFGS) |
| Evaluation count | 521 unique dev posts; all five saved folds |
| Primary metric | Pooled macro-F1 **0.6993617572733752**, higher is better, unitless 0-1 |
| Accuracy | 0.8522072936660269 |
| Fold macro-F1 mean / sample SD | 0.7002228948449116 / 0.05880184131073632 |
| Negative / neutral / positive F1 | 0.8106060606 / 0.3913043478 / 0.8961748634 |
| Runtime | 0.8973 seconds, excluding imports and output writing |
| Hardware | CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD |
| Protected test outcomes | Zero test posts scored |

[Metrics](b1_logreg_metrics.json) contain pooled
and per-fold metrics, evaluation count, exact configuration, data/split identifiers,
hardware, runtime, and output paths.
[Predictions](b1_logreg_predictions.csv)
contain ID, fold, correct label, and prediction for all 521 dev posts.

The saved CSV was checked against all 521 development IDs and their saved folds,
then rescored with the shared metrics. Every ID occurs once, and the recomputed
metrics match the results file.

The neutral slice remains a limitation: 9 of 32 neutral dev posts were correct;
22 were classified positive and 1 negative.

### B1 rerun

Run `python run_all.py --skip-b2` or `python src/baseline_logreg.py` from the
repository root. Expected: 521 dev predictions, pooled macro-F1 0.6993617573,
accuracy 0.8522072937, and all five fits converged. A fresh teammate rerun remains
pending. Record the teammate's name, date, environment, command, and observed
results. Compare predictions by ID and explain differences.

B1 reports use the shared `cv_report` format (`pooled` and per-fold fields). The
shared loader reads the cleaned file and returns dev rows; only those returned
rows are used for training and scoring.

Current B1 reports retain the assignment's required runtime, hardware, run/data/split
identifiers, configuration, evaluation count, metric definition, and output paths.
File hashes, package-version dumps, and per-fold timing/iteration records were
removed from the runner. Older B1 output files were removed; the links above
point to the retained current run.

The baseline name and run-ID prefix are `baseline_logreg`. The recorded run
timestamp, predictions, scores, and runtime are preserved.
