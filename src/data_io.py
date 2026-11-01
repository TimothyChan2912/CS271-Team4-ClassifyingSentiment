"""Load the development partition for experiments.

Enforces the test-partition rule (splits/README.md): test rows are filtered out
of the split record before any text or label is joined in, so experiment code
never holds test posts in memory.
"""
import pandas as pd

CLEAN = "data/processed/clean.parquet"
SPLIT = "splits/split.csv"
EXPECTED_DEV = 521


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
