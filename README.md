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
Download the dataset from
[AmaanP314/youtube-comment-sentiment on Hugging Face](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment).
The active input is `data/raw/youtube-comments-sentiment.csv`.

On **Linux and macOS**, run the same commands from the repository root to download
the CSV directly to the expected path before running the pipeline:

```bash
mkdir -p data/raw
curl -fL "https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment/resolve/main/youtube-comments-sentiment.csv" \
  -o data/raw/youtube-comments-sentiment.csv
```

Required columns are
`CommentID`, `CommentText`, and `Sentiment`; video, author, and
engagement metadata are excluded from model inputs.

`--datasize` is a positive integer limiting the **first N CSV records read before
cleaning**, including records with multiline comments. It defaults to `10000`.
If the file has fewer records, all available records are used. The final dataset
can be smaller after filtering and deduplication.

The cleaner trims comment whitespace and standardizes the existing positive,
negative, and neutral labels. It removes empty comments (including URL/mention-only
comments), missing/unsupported labels, normalized duplicates, and all copies of
texts with conflicting labels. Duplicate comparison ignores case, URLs, mentions,
and repeated whitespace; the retained model text preserves casing, punctuation,
URLs, mentions, and emojis. The old emotion map is not used.

Cleaned data is saved as Apache Parquet at `data/processed/clean.parquet` with
columns `id`, `text`, and `sentiment`. IDs are `youtube_<CommentID>`.
The raw CSV is unchanged. `evidence/prepare_data_counts.json` records the
requested limit, rows actually read, cleaning counts, class counts, and the
full raw-file SHA-256.

# Baseline Run Routes
Run everything from the repository root.

```bash
python run_all.py                       # first 10,000 raw comments
python run_all.py --datasize 20000      # first 20,000 raw comments
python run_all.py --datasize 1000 --skip-b2  # smaller CPU-only baseline run
```

Every run cleans the selected input, regenerates and verifies the deterministic
80/20 dev/test split, then runs the baselines and their emoji ablations. Development uses five stratified
folds. A sample too small to represent all three classes in the split/folds stops
with a request to increase `--datasize`. `--remake-split` is still accepted
for compatibility, but split regeneration is now automatic.

| Step | Command | Output |
|---|---|---|
| Clean data | `python src/data_cleaning.py --datasize 10000` | `data/processed/clean.parquet`, `evidence/prepare_data_counts.json` |
| Make split (after every cleaning run) | `python src/make_split.py` | `splits/split.csv`, `evidence/split_summary.json` |
| Check split | `python src/check_split.py > evidence/split_check_output.txt` | `All split checks passed.` |
| B1: logistic regression | `python src/baseline_logreg.py` | `evidence/b1_*` |
| B2: TweetEval RoBERTa, zero-shot | `python src/baseline_tweeteval.py` | `evidence/b2_tweeteval_metrics.json`, `evidence/b2_tweeteval_predictions.csv` |
| B1 emoji ablations | `python src/run_b1_emoji_ablation.py` | `evidence/ablations/b1/*` |
| B2 emoji ablations | `python src/run_b2_emoji_ablation.py` | `evidence/ablations/b2/*` |
| Consolidated report | Automatic at the end of `run_all.py` | `evidence/run_summary.csv`, `evidence/run_summary.json` |

All baselines score the dev partition only, load data through `src/data_io.py`,
and compute metrics through `src/metrics.py`. Row counts are validated against
the generated split rather than fixed dataset sizes.

The committed results, `configs/PROTOCOL.md`, and `evidence/EVIDENCE_RECORD.md`
describe the earlier Social Media Sentiments dataset. They are historical results,
not expected YouTube scores. A new run replaces the evidence files for the steps
it executes; rerun any additional experiments before comparing their results on
the new dataset.

## Consolidated results

For development and held-out evaluation results across all models and emoji
conditions, run:

```bash
python run_all.py --datasize 10000 --proposed --final
```

`evidence/run_summary.csv` contains nine rows: B1, B2, and proposed, each
with `raw` (emojis kept), `removed`, and `described` (English emoji names).
The companion JSON includes the same rows, grouped class counts, the selected
learning rate/seeds, input provenance, and metric definitions.

Each row includes:

- `dev_positive`, `dev_negative`, `dev_neutral`, `dev_total`, and `dev_emoji_posts`.
- The corresponding `test_*` counts for the held-out evaluation partition.
- `dev_macro_f1` and `test_macro_f1`.
- `dev_accuracy` and `test_accuracy` across all posts, including neutral posts.
- `dev_emoji_accuracy` and `test_emoji_accuracy` restricted to posts that contained
  emojis in the **original cleaned text**, even for removed/described conditions.
- `dev_status`, `test_status`, and run counts so missing results are explicit.

All scores range from 0 to 1. Development scores pool the out-of-fold predictions.
Proposed-model scores average the per-seed metrics at the selected learning rate;
class and emoji counts count each post once. An emoji subset with zero posts has
blank CSV/null JSON accuracy. Saved predictions must match the current partition's
IDs, labels, and folds; incomplete seed sets are rejected.

Without `--final`, test metrics remain blank while evaluation class counts are
still included. Without `--proposed`, proposed results remain blank unless the
final-training route is requested. `--skip-b2` leaves B2 results blank and skips
its ablations. Final evaluation now evaluates both baseline ablations as well.
Rebuild the report from existing predictions with `python src/run_summary.py`,
adding the same `--proposed` / `--skip-b2` flags; to include test results, add
`--final` and set `FINAL_EVAL=1`.


## B1 implementation and reproduction

The B1 entry point listed above, `src/baseline_logreg.py`, runs the scikit-learn
TF-IDF/logistic-regression baseline. It uses the existing `load_dev` and
`cv_report` functions. `python run_all.py --skip-b2` runs data preparation,
split generation and checks, and B1. B1 was also verified with Python 3.13.7 on Windows.

Run the focused checks with `python -m unittest discover -s tests -v`.
The fixed settings are in `configs/baseline_logreg.json`.
A fresh pipeline is fitted on four dev folds and scored on the fifth for each
of the five saved folds. A convergence warning stops the run.

B1 writes directly to `evidence/`, producing `b1_logreg_predictions.csv`
(one held-out prediction per dev comment, with `p_negative`, `p_neutral`, and `p_positive`)
and `b1_logreg_metrics.json` (shared
pooled/per-fold metrics, evaluation count, configuration, data/split identifiers,
runtime, CPU information, and output paths). Each run replaces these two files,
matching B2's output layout. Reports use B2's `pooled` and `per_fold_macro_f1*` fields.

The additional `binary_pos_neg_vs_B0` diagnostic uses the same shared scorer as
B2. It excludes true-neutral posts, compares the positive and negative
probabilities (ties predict negative), and reports accuracy over the positive
and negative development comments. The three-class predictions and primary macro-F1 remain unchanged.
The Stanford 83% result is historical context on different data, not a directly
comparable benchmark. `python src/baseline_logreg.py` regenerates the predictions,
probabilities, and metrics from the five saved development folds.

Loader clarification: `load_dev` reads the cleaned Parquet file internally,
then returns the dev rows. Only those returned rows enter B1 training and scoring.
