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
- **Expected values:** dev 521 posts, 131 test posts; seed 42, 5 folds, test fraction 0.2. Fill in from the file. A rerun must produce an identical `split.csv`.

## 5. Split Boundary Check (`evidence/split_check_output.txt`)

- **Produced by:** `python src/check_split.py > evidence/split_check_output.txt`.
- **What it checks:** split IDs match the cleaned data and contain no duplicates; no post is in both dev and test; no identical text appears in both dev and test; all 5 folds exist; every fold contains all three classes; test posts all have fold -1.
- **Expected output:** `All split checks passed.` If any check fails, the script stops with an assertion message naming the problem, and the split must not be used.
