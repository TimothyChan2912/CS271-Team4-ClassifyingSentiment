# splits/
 
This folder holds the saved record of which post belongs to which partition. It is the source of truth for every experiment. Do not edit files here by hand.
 
## Files
 
| File | Contents |
|---|---|
| `split.csv` | One row per post, with columns `id`, `partition`, `fold` |
 
### Columns
 
| Column | Values | Meaning |
|---|---|---|
| `id` | `youtube_<CommentID>` | Source comment ID from `data/processed/clean.parquet` |
| `partition` | `dev` or `test` | `dev` posts are used for development; `test` posts are protected until the final evaluation |
| `fold` | `0` to `4` for dev posts, `-1` for test posts | Cross-validation fold in which the post is held out |
 
The file contains IDs and assignments only, with no post text and no labels. Join it to `data/processed/clean.parquet` on `id` to get text and labels.
 
## How it was made
 
```bash
python src/data_cleaning.py --datasize 10000  # limits raw CSV records before cleaning
python src/make_split.py        # writes splits/split.csv and evidence/split_summary.json
python src/check_split.py > evidence/split_check_output.txt
```
 
| Setting | Value |
|---|---|
| Seed | 42 |
| Test partition | Stratified 20% on `sentiment` |
| Development partition | Remaining 80%, stratified 5-fold cross-validation on `sentiment` |
| Ordering | Records sorted by `id` before splitting |
 
`run_all.py` always regenerates and verifies this split after cleaning.
The earlier social-media experiment protocol is preserved in `configs/PROTOCOL.md`.
 
## Data version this split depends on
 
The active input is `data/raw/youtube-comments-sentiment.csv`. IDs derive from
the source `CommentID` and stay stable for retained comments when the input
limit changes. Partition and fold assignments depend on the complete selected
dataset, so changing `--datasize`, the CSV, or cleaning rules requires a new split.

`evidence/prepare_data_counts.json` records the input hash, requested limit,
and final size. `evidence/split_summary.json` records the actual partition
and class counts. The loaders reject splits whose IDs differ from the cleaned data.

The committed `social_*` split is a historical record from the previous dataset;
it is replaced when the YouTube pipeline runs.
 
## Verifying a rerun
 
1. Regenerate the split with `python src/make_split.py`.
2. Run `python src/check_split.py` to check ID coverage, duplicate leakage,
   class coverage, and deterministic reconstruction.
3. Identical input files and `--datasize` values should produce identical split files.

## Test partition rule
 
Experiment scripts must not load rows with `partition == test` until the final evaluation. Scripts for baselines should filter to `partition == dev` only.
