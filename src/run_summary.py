"""Write one comparison table for development/test results and emoji ablations.

Run after the corresponding experiments, using the same flags as run_all.py:
    python src/run_summary.py --proposed --final
"""
import argparse
import json
from pathlib import Path

import emoji
import numpy as np
import pandas as pd

if __package__:
    from .data_io import load_dev, load_test
    from .metrics import LABELS, align_predictions, score
else:
    from data_io import load_dev, load_test
    from metrics import LABELS, align_predictions, score

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MODES = ("raw", "removed", "described")
MODELS = {"B1": "TF-IDF + logistic regression", "B2": "TweetEval zero-shot",
          "P": "Fine-tuned RoBERTa"}


def emoji_mask(posts):
    # Always inspect the original cleaned text, before removing/describing emojis.
    return posts["text"].map(lambda text: emoji.emoji_count(text) > 0).to_numpy()


def prediction_paths(evidence, model, mode, partition, selection=None):
    evidence = Path(evidence)
    if model == "P":
        if selection is None:
            return []
        lr, seeds = selection
        if partition == "test":
            return [evidence / "final" / f"P_{mode}_seed{seed}_test_predictions.csv"
                    for seed in seeds]
        folder = "cv" if mode == "raw" else "ablation"
        return [evidence / "proposed" / folder / f"P_{mode}_lr{lr:g}_seed{seed}_predictions.csv"
                for seed in seeds]
    if partition == "test":
        name = model if mode == "raw" else f"{model}_{mode}"
        return [evidence / "final" / f"{name}_test_predictions.csv"]
    if mode != "raw":
        return [evidence / "ablations" / model.lower() / f"emoji-{mode}_predictions.csv"]
    name = "b1_logreg" if model == "B1" else "b2_tweeteval"
    return [evidence / f"{name}_predictions.csv"]


def summarize_predictions(paths, posts, enabled=True):
    result = {"status": "not_run", "runs": 0, "macro_f1": None,
              "accuracy": None, "emoji_accuracy": None}
    if not enabled or posts is None or not paths:
        return result
    present = [path.exists() for path in paths]
    if not any(present):
        return result
    if not all(present):
        missing = [str(path) for path, exists in zip(paths, present) if not exists]
        raise ValueError(f"Incomplete seed results; refusing to average a subset: {missing}")
    mask = emoji_mask(posts)
    metrics = []
    for path in paths:
        try:
            preds = align_predictions(pd.read_csv(path), posts)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        measured = score(preds.y_true, preds.y_pred)
        measured["emoji_accuracy"] = (
            float((preds.y_true.to_numpy()[mask] == preds.y_pred.to_numpy()[mask]).mean())
            if mask.any() else None)
        metrics.append(measured)
    result.update(status="complete", runs=len(metrics))
    for key in ("macro_f1", "accuracy", "emoji_accuracy"):
        result[key] = (float(np.mean([m[key] for m in metrics]))
                       if metrics[0][key] is not None else None)
    return result


def build_report(dev, test, split_summary, evidence, include_proposed=False, include_b2=True):
    """Use pooled dev predictions; average each metric across proposed-model seeds."""
    evidence = Path(evidence)
    counts = {}
    for partition, posts in (("dev", dev), ("test", test)):
        labels = split_summary["class_counts_by_partition"][partition]
        count = {label: int(labels.get(label, 0)) for label in LABELS}
        count["total"] = int(split_summary["partition_sizes"][partition])
        if sum(count[label] for label in LABELS) != count["total"]:
            raise ValueError(f"Inconsistent {partition} counts in split_summary.json")
        count["emoji_posts"] = split_summary.get("emoji_counts_by_partition", {}).get(partition)
        if posts is not None:
            actual = posts.sentiment.value_counts()
            if len(posts) != count["total"] or any(
                    int(actual.get(label, 0)) != count[label] for label in LABELS):
                raise ValueError(f"{partition} counts do not match the loaded dataset")
            count["emoji_posts"] = int(emoji_mask(posts).sum())
        counts[partition] = count

    selection = None
    grid_path = evidence / "proposed" / "cv_grid.json"
    if include_proposed and grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        lr = float(grid["selected_learning_rate"])
        seeds = grid["grid"][f"{lr:g}"]["seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("The selected learning rate must have a nonempty, unique seed list.")
        selection = lr, seeds

    rows = []
    for model, name in MODELS.items():
        enabled = (model != "P" or include_proposed) and (model != "B2" or include_b2)
        for mode in MODES:
            row = {"model": model, "model_name": name, "emoji_mode": mode}
            for partition, posts in (("dev", dev), ("test", test)):
                row.update({f"{partition}_{key}": value for key, value in counts[partition].items()})
                paths = prediction_paths(evidence, model, mode, partition, selection)
                metrics = summarize_predictions(paths, posts, enabled=enabled)
                row.update({f"{partition}_{key}": value for key, value in metrics.items()})
            rows.append(row)
    return {
        "class_counts": counts,
        "selected_learning_rate": selection[0] if selection else None,
        "proposed_seeds": selection[1] if selection else [],
        "notes": [
            "Development counts refer to unique training/validation posts, not a sum over folds.",
            "Dev macro-F1 and accuracy use pooled out-of-fold predictions; B2 is zero-shot.",
            "Test metrics use the held-out evaluation partition.",
            "Macro-F1 and accuracy include negative, neutral, and positive labels.",
            "Emoji accuracy uses posts with emojis in the original cleaned text for every variant.",
            "Proposed metrics are arithmetic means of per-seed metrics at the selected learning rate.",
            "Metrics range from 0 to 1. Missing runs are null/blank; zero emoji posts means null accuracy.",
        ],
        "results": rows,
    }


def write_report(report, evidence=EVIDENCE):
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = evidence / "run_summary.csv", evidence / "run_summary.json"
    pd.DataFrame(report["results"]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed", action="store_true", help="include saved proposed-model results")
    parser.add_argument("--final", action="store_true", help="include held-out test results (FINAL_EVAL=1)")
    parser.add_argument("--skip-b2", action="store_true")
    args = parser.parse_args(argv)
    split_summary = json.loads((EVIDENCE / "split_summary.json").read_text(encoding="utf-8"))
    dev = load_dev()
    test = load_test() if args.final else None
    report = build_report(dev, test, split_summary, EVIDENCE,
                          include_proposed=args.proposed, include_b2=not args.skip_b2)
    report["data_preparation"] = json.loads(
        (EVIDENCE / "prepare_data_counts.json").read_text(encoding="utf-8"))
    for path in write_report(report, EVIDENCE):
        print(f"Saved {path}")
    return report


if __name__ == "__main__":
    main()
