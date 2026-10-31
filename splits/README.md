# splits/
 
This folder holds the saved record of which post belongs to which partition. It is the source of truth for every experiment. Do not edit files here by hand.
 
## Files
 
| File | Contents |
|---|---|
| `split.csv` | One row per post, with columns `id`, `partition`, `fold` |
 
### Columns
 
| Column | Values | Meaning |
|---|---|---|
| `id` | `social_0000` to `social_0651` | Stable post ID from `data/processed/clean.parquet` |
| `partition` | `dev` or `test` | `dev` posts are used for development; `test` posts are protected until the final evaluation |
| `fold` | `0` to `4` for dev posts, `-1` for test posts | Cross-validation fold in which the post is held out |
 
The file contains IDs and assignments only, with no post text and no labels. Join it to `data/processed/clean.parquet` on `id` to get text and labels.
 
## How it was made
 
```bash
python src/data_cleaning.py     # writes data/processed/clean.parquet
python src/make_split.py        # writes splits/split_v1.csv and evidence/split_summary.json
python src/check_split.py > evidence/split_check_output.txt
```
 
| Setting | Value |
|---|---|
| Seed | 42 |
| Test partition | Stratified 20% on `sentiment` |
| Development partition | Remaining 80%, stratified 5-fold cross-validation on `sentiment` |
| Ordering | Records sorted by `id` before splitting |
 
The full protocol is in `PROTOCOL.md`.
 
## Data version this split depends on
 
The IDs are assigned after sorting by (text, emotion), so they only match if the cleaning step produced the same data. Before using or regenerating the split, confirm:
 
| Check | Expected |
|---|---|
| Raw file SHA-256 | `725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b` |
| Rows in `clean.parquet` | 652 |
| Class counts | positive 431, negative 181, neutral 40 |
| Label map | `configs/emotion_map.csv` unchanged |
 
Changing the label map or the cleaning rules changes the IDs and the row count, which invalidates this split.
 
## Verifying a rerun
 
1. Regenerate the split with `python src/make_split.py`.
2. Compare it to the committed file, for example with `git diff splits/split.csv` (no output means identical) or by comparing hashes.
3. Record the expected and observed results in `evidence/EVIDENCE_RECORD.md`.

## Test partition rule
 
Experiment scripts must not load rows with `partition == test` until the final evaluation. Scripts for baselines should filter to `partition == dev` only.