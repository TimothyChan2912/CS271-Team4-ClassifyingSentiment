"""Build evidence/results.csv: one results row per baseline run.

Reads every evidence/b*_metrics*.json written through metrics.cv_report and adds
the data/split version, so each row carries the fields the assignment requires:
baseline/run ID, data and split version, configuration, seed, evaluation count,
primary metric (name, value, direction, units), runtime, hardware, output paths.
"""
import glob, hashlib, json, os

import pandas as pd


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


raw_sha = json.load(open("evidence/prepare_data_counts.json"))["raw_sha256"]
split_sha = sha256("splits/split.csv")

rows = []
for path in sorted(glob.glob("evidence/b*_metrics*.json")):
    r = json.load(open(path))
    cfg = r.get("config", {})
    pooled = r["pooled"]
    run_id = os.path.basename(path).replace("_metrics", "").replace(".json", "")
    pred_path = path.replace("_metrics", "_predictions").replace(".json", ".csv")
    rows.append({
        "baseline_id": cfg.get("id", run_id.split("_")[0].upper()),
        "run_id": run_id,
        "raw_data_sha256": raw_sha[:16],
        "split_csv_sha256": split_sha[:16],
        "partition": "dev, 5-fold out-of-fold",
        "configuration": json.dumps({k: v for k, v in cfg.items()
                                     if k not in ("versions", "hardware", "runtime_seconds")},
                                    sort_keys=True),
        "seed": cfg.get("seed", "n/a"),
        "n_eval": pooled["n"],
        "primary_metric": "macro-F1, pooled out-of-fold",
        "value": round(pooled["macro_f1"], 4),
        "direction": "higher is better",
        "units": "unitless, 0 to 1",
        "per_fold_macro_f1_mean": round(r.get("per_fold_macro_f1_mean", float("nan")), 4),
        "per_fold_macro_f1_std": round(r.get("per_fold_macro_f1_std", float("nan")), 4),
        "accuracy": round(pooled["accuracy"], 4),
        **{f"f1_{k}": round(v, 4) for k, v in pooled["per_class_f1"].items()},
        "binary_pos_neg_accuracy": r.get("binary_pos_neg_vs_B0", {}).get("accuracy"),
        "runtime_seconds": cfg.get("runtime_seconds"),
        "hardware": cfg.get("hardware"),
        "outputs": path + ("; " + pred_path if os.path.exists(pred_path) else ""),
    })

assert rows, "no evidence/b*_metrics*.json files found"
df = pd.DataFrame(rows)
df.to_csv("evidence/results.csv", index=False)
print(df[["baseline_id", "run_id", "n_eval", "value", "per_fold_macro_f1_mean",
          "per_fold_macro_f1_std", "accuracy", "f1_neutral"]].to_string(index=False))
