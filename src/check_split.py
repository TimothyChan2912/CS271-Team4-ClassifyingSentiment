# src/check_split.py
import pandas as pd

from make_split import build_split


def normalize(s):
    # must match the duplicate-detection rule in data_cleaning.py
    return (s.str.lower()
            .str.replace(r"http\S+|www\.\S+", "", regex=True)
            .str.replace(r"@\w+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


df = pd.read_parquet("data/processed/clean.parquet")
sp = pd.read_csv("splits/split.csv")
m = sp.merge(df, on="id")
dev, test = m[m.partition == "dev"], m[m.partition == "test"]

assert len(sp) == len(df) and sp["id"].is_unique, "ids missing or duplicated"
assert set(sp["id"]) == set(df["id"]), "split ids do not match clean data"
assert set(dev["id"]).isdisjoint(test["id"]), "dev/test overlap"
assert set(dev["fold"]) == {0, 1, 2, 3, 4}, "folds missing"
assert (test["fold"] == -1).all(), "test posts must have fold -1"
# no text shared between dev and test, exact or after duplicate normalization
assert set(dev["text"]).isdisjoint(test["text"]), "identical text in dev and test"
assert set(normalize(dev["text"])).isdisjoint(normalize(test["text"])), \
    "near-duplicate (normalized) text in dev and test"
# every fold has all three classes
for k, g in dev.groupby("fold"):
    assert g["sentiment"].nunique() == 3, f"fold {k} is missing a class"
# determinism: rebuilding from clean.parquet reproduces the saved split exactly
rebuilt, _ = build_split(df)
rebuilt = rebuilt[["id", "partition", "fold"]].sort_values("id").reset_index(drop=True)
saved = sp.sort_values("id").reset_index(drop=True)
assert rebuilt.equals(saved), "rebuilt split differs from splits/split.csv"
print("All split checks passed.")
