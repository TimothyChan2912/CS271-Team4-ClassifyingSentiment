import os, json, hashlib
import pandas as pd
import emoji

RAW = "data/raw/sentimentdataset.csv"
MAP_PATH = "configs/emotion_map.csv"
DRAFT_PATH = "configs/emotion_map_draft.csv"

for d in ["configs", "evidence", "data/processed"]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv(RAW)
df = df.rename(columns={"Text": "text", "Sentiment": "emotion"})
df = df[["text", "emotion"]].copy()
counts = {"raw": len(df)}

# basic cleaning
df["text"] = df["text"].astype(str).str.strip()
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
df = df[df["text"].ne("") & df["text"].ne("nan")]
counts["after_empty_removed"] = len(df)

# duplicates and conflicts
df["norm"] = (df["text"].str.lower()
              .str.replace(r"http\S+|www\.\S+", "", regex=True)
              .str.replace(r"@\w+", "", regex=True)
              .str.replace(r"\s+", " ", regex=True).str.strip())

conflicts = df.groupby("norm")["emotion"].nunique()
counts["conflicting_groups"] = int((conflicts > 1).sum())
df = df[~df["norm"].isin(conflicts[conflicts > 1].index)]
df = df.drop_duplicates(subset="norm")
counts["after_dedup"] = len(df)

# stage 1: write the draft map and stop if the real map doesn't exist yet
if not os.path.exists(MAP_PATH):
    emo = df["emotion"].value_counts().rename_axis("emotion").reset_index(name="n")
    emo["sentiment"] = ""   # fill with positive / negative / neutral / drop
    emo.to_csv(DRAFT_PATH, index=False)
    raise SystemExit(f"Fill in {DRAFT_PATH}, save it as {MAP_PATH}, then rerun.")

# stage 2: apply the map
m = pd.read_csv(MAP_PATH)
m["emotion"] = m["emotion"].str.strip().str.lower()
m["sentiment"] = m["sentiment"].str.strip().str.lower()
m = m.set_index("emotion")["sentiment"]

df["sentiment"] = df["emotion"].map(m)
unmapped = df[df["sentiment"].isna()]["emotion"].unique()
assert len(unmapped) == 0, f"Unmapped emotions: {unmapped}"

counts["dropped_ambiguous"] = int((df["sentiment"] == "drop").sum())
df = df[df["sentiment"].isin(["negative", "neutral", "positive"])]
counts["final"] = len(df)

# stable IDs and save
df = df.sort_values(["text", "emotion"]).reset_index(drop=True)
df["id"] = [f"social_{i:04d}" for i in df.index]
out = df[["id", "text", "emotion", "sentiment"]]
out.to_parquet("data/processed/clean.parquet", index=False)

# evidence
counts["class_counts"] = out["sentiment"].value_counts().to_dict()
counts["posts_with_emoji"] = int(out["text"].map(lambda t: emoji.emoji_count(t) > 0).sum())
counts["raw_sha256"] = hashlib.sha256(open(RAW, "rb").read()).hexdigest()
with open("evidence/prepare_data_counts.json", "w") as f:
    json.dump(counts, f, indent=2)
print(json.dumps(counts, indent=2))