import os, json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED, N_FOLDS, TEST_SIZE = 42, 5, 0.2
os.makedirs("splits", exist_ok=True)
os.makedirs("evidence", exist_ok=True)

df = pd.read_parquet("data/processed/clean.parquet").sort_values("id").reset_index(drop=True)

dev, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["sentiment"])
dev = dev.sort_values("id").reset_index(drop=True)
test = test.sort_values("id").reset_index(drop=True)

dev["fold"] = -1
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (_, val_idx) in enumerate(skf.split(dev, dev["sentiment"])):
    dev.loc[val_idx, "fold"] = k

dev["partition"] = "dev"
test["partition"], test["fold"] = "test", -1
split = pd.concat([dev, test]).sort_values("id")
split[["id", "partition", "fold"]].to_csv("splits/split_v1.csv", index=False)

# counts for the paper
summary = {"partition_sizes": split["partition"].value_counts().to_dict()}
summary["class_counts_by_partition"] = (
    split.groupby("partition")["sentiment"].value_counts().unstack(fill_value=0).to_dict("index"))
summary["class_counts_by_fold"] = (
    dev.groupby("fold")["sentiment"].value_counts().unstack(fill_value=0).to_dict("index"))
summary["settings"] = {"seed": SEED, "n_folds": N_FOLDS, "test_size": TEST_SIZE}
with open("evidence/split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))