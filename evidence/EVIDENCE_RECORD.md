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