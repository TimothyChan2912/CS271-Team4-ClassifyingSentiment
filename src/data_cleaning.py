"""Prepare labeled YouTube comments for the sentiment models.

--datasize limits CSV records read, before filtering and deduplication.
"""
import argparse
import hashlib
import json
from pathlib import Path

import emoji
import pandas as pd

RAW = "data/raw/youtube-comments-sentiment.csv"
DEFAULT_DATASIZE = 10000
SENTIMENTS = {"negative", "neutral", "positive"}


def normalize_text(text):
    """Comparison keys only; the model still receives the original trimmed text."""
    return (text.str.lower()
            .str.replace(r"http\S+|www\.\S+", "", regex=True)
            .str.replace(r"@\w+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def clean_comments(raw):
    """Return the model table and cleaning counts for a YouTube CSV frame."""
    columns = {"CommentID": "id", "CommentText": "text", "Sentiment": "sentiment"}
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f"YouTube CSV is missing required columns: {sorted(missing)}")
    df = raw[list(columns)].rename(columns=columns).copy()
    counts = {"raw": len(df)}

    df["id"] = df["id"].astype("string").fillna("").str.strip()
    df["text"] = df["text"].astype("string").fillna("").str.strip()
    df["sentiment"] = df["sentiment"].astype("string").fillna("").str.strip().str.lower()
    df["norm"] = normalize_text(df["text"])
    # Also exclude comments consisting entirely of URLs or mentions.
    df = df[df["norm"].ne("")].copy()
    counts["after_empty_removed"] = len(df)

    # This source already supplies three-class labels; no emotion map is needed.
    valid = df["sentiment"].isin(SENTIMENTS)
    counts["dropped_invalid_sentiment"] = int((~valid).sum())
    df = df[valid].copy()

    conflicts = df.groupby("norm")["sentiment"].nunique()
    counts["conflicting_groups"] = int((conflicts > 1).sum())
    conflicting = df["norm"].isin(conflicts[conflicts > 1].index)
    counts["conflicting_rows"] = int(conflicting.sum())
    df = df[~conflicting]
    counts["duplicates_removed"] = int(df.duplicated(subset="norm").sum())
    df = df.drop_duplicates(subset="norm")
    counts["after_dedup"] = len(df)

    if df.empty:
        raise ValueError("No usable YouTube comments remain after cleaning; increase --datasize.")
    if df["id"].eq("").any() or not df["id"].is_unique:
        raise ValueError("Cleaned comments must have nonempty, unique CommentID values.")

    # Source IDs remain stable when --datasize changes.
    df["id"] = "youtube_" + df["id"]
    out = df[["id", "text", "sentiment"]].sort_values("id").reset_index(drop=True)
    counts["final"] = len(out)
    counts["class_counts"] = out["sentiment"].value_counts().to_dict()
    counts["posts_with_emoji"] = int(out["text"].map(lambda t: emoji.emoji_count(t) > 0).sum())
    return out, counts


def prepare_data(raw_path=RAW, datasize=DEFAULT_DATASIZE):
    if datasize <= 0:
        raise ValueError("--datasize must be a positive integer.")
    raw = pd.read_csv(raw_path, usecols=["CommentID", "CommentText", "Sentiment"],
                      dtype="string", nrows=datasize)
    out, counts = clean_comments(raw)
    # Hash in a stream instead of loading the entire large CSV into memory.
    with open(raw_path, "rb") as source:
        raw_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    counts.update({"raw_file": str(raw_path), "raw_sha256": raw_sha256,
                   "datasize_requested": datasize})
    return out, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasize", type=int, default=DEFAULT_DATASIZE,
                        help="maximum raw CSV records to read before cleaning (default: 10000)")
    args = parser.parse_args(argv)
    if args.datasize <= 0:
        parser.error("--datasize must be a positive integer")
    out, counts = prepare_data(datasize=args.datasize)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("evidence").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/processed/clean.parquet", index=False)
    Path("evidence/prepare_data_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
