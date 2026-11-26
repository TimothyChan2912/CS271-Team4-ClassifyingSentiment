"""Load the development or test partition for experiments.

`load_dev` returns only development rows to experiment code. The cleaned
Parquet file is read internally to validate the saved split against its IDs.
`load_test` refuses to return test posts unless FINAL_EVAL=1 is set.
"""
import os

import pandas as pd

CLEAN = "data/processed/clean.parquet"
SPLIT = "splits/split.csv"


def _load_partition(partition):
    split = pd.read_csv(SPLIT)
    assert split["id"].notna().all() and split["id"].is_unique, "split IDs are missing or duplicated"
    assert set(split["partition"]) == {"dev", "test"}, "invalid split partitions"
    ids = split[split["partition"] == partition][["id", "fold"]]
    clean = pd.read_parquet(CLEAN, columns=["id", "text", "sentiment"])
    rows = ids.merge(clean, on="id", how="inner", validate="one_to_one")
    assert len(split) == len(clean) and set(split["id"]) == set(clean["id"]), (
        "clean.parquet and split.csv have different IDs; rerun src/make_split.py")
    assert len(rows) == len(ids) and len(rows) > 0, f"missing {partition} posts in clean.parquet"
    return rows.sort_values("id").reset_index(drop=True)


def load_dev():
    dev = _load_partition("dev")
    assert set(dev["fold"]) == {0, 1, 2, 3, 4}
    return dev


def load_test():
    if os.environ.get("FINAL_EVAL") != "1":
        raise RuntimeError(
            "Test posts are protected. Set FINAL_EVAL=1 only for the single final "
            "evaluation, after every setting has been chosen on development data.")
    test = _load_partition("test")
    assert test["fold"].eq(-1).all(), "test posts must have fold -1"
    return test.drop(columns="fold")
