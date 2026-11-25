"""Load the development or test partition for experiments.

Enforces the test-partition rule (splits/README.md): `load_dev` filters test
rows out of the split record before any text or label is joined in, so
development code never holds test posts in memory. `load_test` is used only by
the final evaluation and refuses to run unless FINAL_EVAL=1 is set, so test
posts cannot be read by accident while settings are still being chosen.
"""
import os

import pandas as pd

CLEAN = "data/processed/clean.parquet"
SPLIT = "splits/split.csv"
EXPECTED_DEV = 521
EXPECTED_TEST = 131


def load_dev():
    split = pd.read_csv(SPLIT)
    dev_ids = split[split["partition"] == "dev"][["id", "fold"]]
    clean = pd.read_parquet(CLEAN, columns=["id", "text", "sentiment"])
    dev = dev_ids.merge(clean, on="id", how="inner", validate="one_to_one")
    assert len(dev) == EXPECTED_DEV, (
        f"expected {EXPECTED_DEV} dev posts, got {len(dev)}; "
        "clean.parquet or split.csv does not match the frozen data version")
    assert set(dev["fold"]) == {0, 1, 2, 3, 4}
    return dev.sort_values("id").reset_index(drop=True)


def load_test():
    if os.environ.get("FINAL_EVAL") != "1":
        raise RuntimeError(
            "Test posts are protected. Set FINAL_EVAL=1 only for the single final "
            "evaluation, after every setting has been chosen on development data.")
    split = pd.read_csv(SPLIT)
    test_ids = split[split["partition"] == "test"][["id"]]
    clean = pd.read_parquet(CLEAN, columns=["id", "text", "sentiment"])
    test = test_ids.merge(clean, on="id", how="inner", validate="one_to_one")
    assert len(test) == EXPECTED_TEST, (
        f"expected {EXPECTED_TEST} test posts, got {len(test)}; "
        "clean.parquet or split.csv does not match the frozen data version")
    return test.sort_values("id").reset_index(drop=True)
