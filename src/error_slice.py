"""One validation error slice across the baseline set: dev posts labeled neutral.

Neutral is the smallest class (32 dev posts) and the weakest for every baseline,
so it is the slice that most limits macro-F1. Uses dev predictions only.
Writes evidence/error_slice_neutral.md.
"""
import glob, os

import pandas as pd

from data_io import load_dev

dev = load_dev()
merged = dev[dev["sentiment"] == "neutral"][["id", "text"]].copy()
n = len(merged)
lines = [f"# Error slice: dev posts whose true label is neutral (n = {n} of {len(dev)})", ""]

names = []
for path in sorted(glob.glob("evidence/b*_predictions.csv")):
    name = os.path.basename(path).split("_predictions")[0]
    pred = pd.read_csv(path, usecols=["id", "y_pred"]).rename(columns={"y_pred": name})
    merged = merged.merge(pred, on="id", how="left")
    names.append(name)
    vc = merged[name].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
    lines.append(f"- **{name}**: {vc['neutral']}/{n} correct (neutral recall {vc['neutral'] / n:.2f}); "
                 f"predicted positive {vc['positive']}, negative {vc['negative']}")

assert names, "no evidence/b*_predictions.csv files found"
wrong_all = merged[(merged[names] != "neutral").all(axis=1)]
lines += ["", f"Neutral posts misclassified by every baseline: {len(wrong_all)} of {n}", "",
          "| id | text (first 120 chars) | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
for _, r in wrong_all.head(10).iterrows():
    text = str(r["text"]).replace("|", "/").replace("\n", " ")[:120]
    lines.append(f"| {r['id']} | {text} | " + " | ".join(str(r[b]) for b in names) + " |")

os.makedirs("evidence", exist_ok=True)
open("evidence/error_slice_neutral.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
