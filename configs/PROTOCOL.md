# Evaluation Protocol (v1)

**Project:** CS271, sentiment classification of social media posts (text and emojis)
**Team:** Timothy Chan, John Brisebois, Shivam Shelar

---

## 1. Task

| Item | Decision |
|---|---|
| Observation unit | One social media post |
| Prediction-time input | Post text only (emojis kept inside the text). No metadata (user, time, platform, likes, hashtag fields) |
| Target | `sentiment` in {negative, neutral, positive} |
| Population | English-language social media posts in the Social Media Sentiments Analysis Dataset (Kaggle) |
| Main question | Does a fine-tuned transformer outperform a classical TF-IDF baseline at 3-class sentiment, and do emojis add information? (the transformer and the emoji ablation are November work) |
| Non-goal | No claim about an author's true emotional state; no summarization of overall sentiment |

**Scope change from the proposal.** The proposal framed the task as emotion classification but specified three classes (positive, negative, neutral) for all models and baselines. The dataset's 191 fine-grained emotion labels are therefore mapped to three sentiment classes (Section 2). Fine-grained emotion and multi-label classification are left as future work.

## 2. Data version and construction

| Item | Value |
|---|---|
| Source | Kaggle: Social Media Sentiments Analysis Dataset (`kashishparmar02/social-media-sentiments-analysis-dataset`) |
| Retrieval date | `10/30/2026` |
| Raw file | `data/raw/sentimentdataset.csv` |
| Raw SHA-256 | `725c6f7bc928c2bf741094f37078024b305982523fe6d0b8cb20a6cf5275c63b` |
| Cleaning script | `src/data_cleaning.py` |
| Label map | `configs/emotion_map.csv` (191 labels mapped; 20 mapped to `drop`) |

**Cleaning stages** (from `evidence/prepare_data_counts.json`):

| Stage | Rows |
|---|---|
| Raw | 732 |
| After removing empty text | 732 |
| After removing exact and normalized duplicates (0 conflicting-label groups) | 706 |
| After dropping labels with no clear polarity (`drop`) | 652 |

**Final class counts:** positive 431, negative 181, neutral 40 (total 652).

**Label mapping rule.** Each emotion maps to its most generic polarity. Labels with mixed, context-dependent, or uninformative polarity (for example nostalgia, ambivalence, bittersweet, and the label "emotion") are excluded rather than forced into a class. `emotion_map.csv` flags 19 labels as judgment calls.

**Input availability.** At prediction time only the post text exists. The original `emotion` column is kept in `data/processed/clean.parquet` for error analysis only and is never a model input.

**Known data issues.** Neutral has only 40 posts. Only 11 posts contain an emoji. 26 duplicate posts were removed. The dataset is small and was collected over a limited window.

## 3. Split

| Item | Decision |
|---|---|
| Split unit | Post (one row per post) |
| Test partition | Stratified 20% of posts, locked until the final evaluation |
| Development partition | Remaining 80%, with stratified 5-fold cross-validation |
| Stratification | On `sentiment` |
| Seed | 42 (both the dev/test split and the fold assignment) |
| Ordering | Records sorted by `id`; IDs assigned after sorting by (text, emotion) |
| Saved record | `splits/split.csv` (columns: `id`, `partition`, `fold`) |
| Generator | `src/make_split.py`; verified by `src/check_split.py` |

**Why cross-validation instead of the proposal's 80/10/10.** With 40 neutral posts, 10% partitions would hold about 4 neutral posts each, making neutral-class F1 mostly noise. Pooling out-of-fold predictions over the development partition gives about 32 neutral examples for scoring.

**Partition sizes and class counts** (from `evidence/split_summary.json`):

| Partition | Total | Negative | Neutral | Positive |
|---|---|---|---|---|
| Dev | `521` | `145` | `32` | `344` |
| Test | `131` | `36` | `8` | `87` |

Per-fold counts are in `evidence/split_summary.json`.

## 4. Partition roles

- **Training folds** fit all learned quantities (vocabulary, IDF weights, model parameters).
- **Held-out dev fold** guides development. Each of the 5 folds is held out once.
- **Test partition** is used once, for the final fixed comparison in November. No selection, tuning, or inspection of test outcomes happens before then.
- **Test exposure so far:** none. 

## 5. Metric and selection rules

| Item | Decision |
|---|---|
| Primary metric | Macro-averaged F1 |
| Direction and units | Higher is better; unitless, range 0 to 1 |
| Label order | [negative, neutral, positive] |
| How it is computed on dev | Pool out-of-fold predictions across all 5 folds and compute macro-F1 once on the pooled predictions. Also report mean and standard deviation of per-fold macro-F1 |
| Secondary metrics | Accuracy; per-class F1 |
| Threshold selection | Not applicable (argmax over class scores) |
| Tuning budget (October) | One fixed configuration per baseline; no sweeps |
| Tuning budget (November, planned) | Small grid chosen on mean CV macro-F1 only |
| Tie rule | If macro-F1 is tied, prefer the simpler model, then the lower-variance one |
| Seeds | Logistic regression is deterministic given the folds. The fine-tuned model will use 3 seeds `42, 43, 44` |
| Completing folds | One baseline evaluation means all 5 folds, never a selected fold |

## 6. Information flow

| Operation | Fit or chosen on | Transformed or scored on |
|---|---|---|
| Duplicate detection (lowercase, strip URLs and @mentions) | No fitting (rule-based, no labels used for removal except conflict check) | Whole dataset, before splitting |
| Label mapping (emotion to sentiment) | Fixed by hand in `configs/emotion_map.csv`, before splitting | Whole dataset |
| TF-IDF vocabulary and IDF weights | Training folds only, refit inside each fold | Held-out fold |
| Logistic regression weights | Training folds only | Held-out fold |
| Regularization strength C | Fixed currently (no selection) | Not applicable |
| TweetEval RoBERTa (zero-shot) | Not fit (pretrained, no training on this data) | All dev posts (equivalent to held-out predictions); test only in November |
| Imputation, scaling, feature selection, threshold selection | Not applicable (no such steps) | Not applicable |

**Note on pre-split steps.** Deduplication and the label map run on the full dataset before splitting. This is intentional: deduplicating after splitting would let near-identical posts land in different partitions. The label map was written from label names alone and not from any model results.

## 7. Baseline roster 

| ID | Method | Role | Configuration | Status |
|---|---|---|---|---|
| B0 | 2009 Stanford study, 83% positive-vs-negative accuracy (Go et al.) | Cited reference only, not rerun. Measured on a different test set, so not directly comparable | n/a | Reference |
| B1 | Logistic regression on TF-IDF unigrams and bigrams | Simple, understandable reference | `C=1.0, min_df=1, class_weight="balanced", max_iter=1000, ngram_range=(1,2), sublinear_tf=True` | Planned: all 5 folds|
| B2 | TweetEval RoBERTa sentiment model (Barbieri et al., 2020), zero-shot | Established comparison from related work | `cardiffnlp/twitter-roberta-base-sentiment, zero-shot, max_length=128, truncation on` | Planned: all dev posts |

**Proposed method (November, not run now):** fine-tuned RoBERTa (DistilRoBERTa as fallback), with the emoji ablation (emojis removed vs. kept as text descriptions).

## 8. Reproducibility checks

- `src/check_split.py` asserts: no id overlap between dev and test, no identical text across dev and test, every fold contains all three classes, split ids match the cleaned data. Output saved to `evidence/split_check_output.txt`.
- Teammate rerun: regenerate the split with `src/make_split.py` and confirm it matches `splits/split.csv` exactly.

## 9. Change policy

This protocol is frozen. Any change requires: a new versioned file, a written explanation, preservation of the old version, and a rerun of every affected comparison. If test results ever influence a decision, that must be disclosed in the paper along with a credible repair.

## 10. Planned limitation

Neutral is a small class, and per-class results for it will be noisy even with pooled out-of-fold scoring. All conclusions are estimates on this dataset and are not claims about every platform or time period.