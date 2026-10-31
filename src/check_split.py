# src/check_split.py
import pandas as pd

df = pd.read_parquet("data/processed/clean.parquet")
sp = pd.read_csv("splits/split.csv")
m = sp.merge(df, on="id")
dev, test = m[m.partition == "dev"], m[m.partition == "test"]

assert len(sp) == len(df) and sp["id"].is_unique, "ids missing or duplicated"
assert set(sp["id"]) == set(df["id"]), "split ids do not match clean data"
assert set(dev["id"]).isdisjoint(test["id"]), "dev/test overlap"
assert set(dev["fold"]) == {0, 1, 2, 3, 4}, "folds missing"
assert (test["fold"] == -1).all()
# no text shared between dev and test
assert set(dev["text"]).isdisjoint(test["text"]), "identical text in dev and test"
# every fold has all three classes
for k, g in dev.groupby("fold"):
    assert g["sentiment"].nunique() == 3, f"fold {k} is missing a class"
print("All split checks passed.")