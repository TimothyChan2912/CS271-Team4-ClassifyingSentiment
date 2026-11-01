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

## 7. Results Table (`evidence/results.csv`)

- **Produced by:** `python src/collect_results.py` (also run at the end of `run_all.py`).
- **Contents:** one row per baseline run with baseline and run ID, raw-data and split-file SHA-256, configuration, seed, evaluation count, primary metric (pooled out-of-fold macro-F1; higher is better; unitless, 0 to 1), per-fold mean and std, accuracy, per-class F1, runtime, hardware, and output paths.
- **Extra configurations tried:** none. One fixed configuration per baseline, as set by protocol v1. The optional `--preprocess` route for B2 was not run.

## 8. Error Slice (`evidence/error_slice_neutral.md`)

- **Produced by:** `python src/error_slice.py` (dev predictions only).
- **Slice:** dev posts whose true label is neutral (32 of 521). Neutral is the smallest class and the lowest per-class F1 for B2, so it limits macro-F1 the most.
- **Reports:** per-baseline neutral recall, where misclassified neutral posts go, and the posts that every baseline gets wrong.

## 9. Teammate Rerun (every baseline)

Run in a fresh clone with a new virtual environment, by a teammate who did not write the baseline being checked:

```bash
python run_all.py --remake-split
git diff --exit-code splits/ evidence/prepare_data_counts.json evidence/split_summary.json && echo "SPLIT REPRODUCED"
```

| Check | Expected | Observed |
|---|---|---|
| Raw SHA-256 | `725c6f7b...5275c63b` | |
| `final` rows; class counts | 652; positive 431, negative 181, neutral 40 | |
| `split.csv` identical to committed file | yes (`SPLIT REPRODUCED`) | |
| `check_split.py` | `All split checks passed.` | |
| B1 pooled macro-F1 (dev, n = 521) | B1_VALUE | |
| B2 pooled macro-F1 (dev, n = 521) | 0.6783 | |
| B2 accuracy; neutral F1 | 0.8196; 0.2500 | |

Small differences in B2 across CPU, Apple GPU (mps), and CUDA can come from floating-point arithmetic; record the device. B1 is deterministic and must match exactly.

Rerun by: ______ on ______ (OS, device, Python version: ______).
