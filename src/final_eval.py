"""Final, one-time evaluation on the protected test partition.

Run only after every setting is chosen on development data:
    FINAL_EVAL=1 python src/proposed_finetune.py final   # P and its ablations
    FINAL_EVAL=1 python src/final_eval.py                # B1, B2, then the combined table
    python src/final_eval.py --aggregate-only            # rebuild the table from saved predictions

B1 is refit on all 521 development posts with its frozen configuration. B2 is the
same zero-shot checkpoint as before. Results go to evidence/final/.

Error bars: 95% bootstrap intervals over test posts (2,000 resamples, seed 0).
For methods with several training seeds, each resample averages macro-F1 across
seeds, so the interval reflects sampling of test posts, and the seed SD is
reported separately. Paired differences use the same resampled posts for both
methods.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

if __package__:
    from .data_io import load_dev, load_test
    from .metrics import LABELS, binary_pos_neg_accuracy, score
else:
    from data_io import load_dev, load_test
    from metrics import LABELS, binary_pos_neg_accuracy, score

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "evidence" / "final"
N_BOOT, BOOT_SEED = 2000, 0
ROLES = {"B1": "simple baseline (TF-IDF + logistic regression)",
         "B2": "zero-shot TweetEval RoBERTa = P without fine-tuning",
         "P-raw": "proposed: fine-tuned RoBERTa, emojis kept",
         "P-removed": "ablation: P with emojis removed",
         "P-described": "ablation: P with emojis as text"}
COMPARISONS = [("P-raw", "B1"), ("P-raw", "B2"), ("P-raw", "P-removed"), ("P-raw", "P-described")]


def save(name, test, probs, meta):
    FINAL.mkdir(parents=True, exist_ok=True)
    preds = test[["id"]].copy()
    preds["fold"], preds["y_true"] = -1, test["sentiment"].values
    preds["y_pred"] = [LABELS[i] for i in probs.argmax(axis=1)]
    for j, lab in enumerate(LABELS):
        preds[f"p_{lab}"] = probs[:, j]
    preds.to_csv(FINAL / f"{name}_test_predictions.csv", index=False)
    result = {**meta, "partition": "test", **score(preds.y_true, preds.y_pred),
              "binary_pos_neg_vs_B0": binary_pos_neg_accuracy(preds.y_true, probs)}
    (FINAL / f"{name}_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: test macro-F1 {result['macro_f1']:.4f} | accuracy {result['accuracy']:.4f}")


def run_b1(dev, test):
    if __package__:
        from .baseline_logreg import CONFIG, make_model
    else:
        from baseline_logreg import CONFIG, make_model
    config = json.loads(CONFIG.read_text())
    model = make_model(config).fit(dev.text, dev.sentiment)
    probs = model.predict_proba(test.text)
    order = [list(model.classes_).index(lab) for lab in LABELS]
    save("B1", test, probs[:, order], {"method_id": "B1", "trained_on": "all 521 dev posts",
                                        "config": config})


def run_b2(test, device):
    if __package__:
        from .baseline_tweeteval import predict
    else:
        from baseline_tweeteval import predict
    b2 = json.loads((ROOT / "evidence" / "b2_tweeteval_metrics.json").read_text())
    revision = b2["config"].get("model_commit") or "main"
    probs, info = predict(test.text.tolist(), 32, device, revision)
    save("B2", test, probs, {"method_id": "B2", "trained_on": "nothing (zero-shot)", **info})


def load_runs():
    """{method_id: [prediction frames, one per seed]} from evidence/final/."""
    runs = {}
    for path in sorted(FINAL.glob("*_test_predictions.csv")):
        stem = path.name.replace("_test_predictions.csv", "")
        if stem in ("B1", "B2"):
            method = stem
        elif m := re.fullmatch(r"P_(raw|removed|described)_seed\d+", stem):
            method = f"P-{m.group(1)}"
        else:
            continue  # e.g. checkpoint re-evaluations, reported separately
        runs.setdefault(method, []).append(pd.read_csv(path).sort_values("id").reset_index(drop=True))
    return runs


def mean_f1(frames, idx=None):
    vals = []
    for f in frames:
        t, p = (f.y_true, f.y_pred) if idx is None else (f.y_true.values[idx], f.y_pred.values[idx])
        vals.append(f1_score(t, p, labels=LABELS, average="macro", zero_division=0))
    return float(np.mean(vals))


def aggregate():
    runs = load_runs()
    assert runs, "no test predictions in evidence/final/"
    n = len(next(iter(runs.values()))[0])
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = {m: np.array([mean_f1(fr, i) for i in boot_idx]) for m, fr in runs.items()}

    rows = []
    for method, frames in runs.items():
        reports = [score(f.y_true, f.y_pred) for f in frames]
        per_seed = [r["macro_f1"] for r in reports]
        binary = [binary_pos_neg_accuracy(f.y_true, f[[f"p_{l}" for l in LABELS]].to_numpy())["accuracy"]
                  for f in frames]
        lo, hi = np.percentile(boots[method], [2.5, 97.5])
        rows.append({"method": method, "role": ROLES.get(method, ""), "n_test": n,
                     "n_seeds": len(frames), "macro_f1": np.mean(per_seed),
                     "seed_sd": np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0,
                     "ci95_low": lo, "ci95_high": hi,
                     "accuracy": np.mean([r["accuracy"] for r in reports]),
                     **{f"f1_{lab}": np.mean([r["per_class_f1"][lab] for r in reports]) for lab in LABELS},
                     "binary_pos_neg_accuracy": np.mean(binary)})
    order = list(ROLES)
    table = pd.DataFrame(rows).sort_values("method", key=lambda s: s.map(order.index)).round(4)

    diffs = []
    for a, b in COMPARISONS:
        if a in runs and b in runs:
            d = boots[a] - boots[b]
            diffs.append({"comparison": f"{a} minus {b}",
                          "macro_f1_difference": round(mean_f1(runs[a]) - mean_f1(runs[b]), 4),
                          "ci95_low": round(float(np.percentile(d, 2.5)), 4),
                          "ci95_high": round(float(np.percentile(d, 97.5)), 4),
                          "share_of_resamples_above_zero": round(float((d > 0).mean()), 3)})
    FINAL.mkdir(parents=True, exist_ok=True)
    table.to_csv(FINAL / "test_results.csv", index=False)
    pd.DataFrame(diffs).to_csv(FINAL / "test_differences.csv", index=False)
    (FINAL / "test_results.json").write_text(json.dumps(
        {"error_bars": f"95% bootstrap over test posts, {N_BOOT} resamples, seed {BOOT_SEED}; "
                       "seed_sd is the SD of macro-F1 across training seeds",
         "results": table.to_dict("records"), "paired_differences": diffs}, indent=2) + "\n")
    print(table[["method", "n_seeds", "macro_f1", "seed_sd", "ci95_low", "ci95_high",
                 "accuracy", "f1_neutral"]].to_string(index=False))
    if diffs:
        print(pd.DataFrame(diffs).to_string(index=False))
    return table, diffs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-b2", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if not args.aggregate_only:
        test = load_test()  # raises unless FINAL_EVAL=1
        run_b1(load_dev(), test)
        if not args.skip_b2:
            run_b2(test, args.device)
    return aggregate()


if __name__ == "__main__":
    main()
