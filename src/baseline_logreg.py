from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

if __package__:
    from .data_io import load_dev
    from .metrics import cv_report
else:
    from data_io import load_dev
    from metrics import cv_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/baseline_logreg.json"


def make_model(config):
    tfidf = dict(config["tfidf"])
    tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    return make_pipeline(TfidfVectorizer(**tfidf), LogisticRegression(**config["classifier"]))


def run_cv(dev, config):
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for fold in range(config["n_folds"]):
            train, validation = dev.loc[dev.fold != fold], dev.loc[dev.fold == fold]
            model = make_model(config).fit(train.text, train.sentiment)
            rows = validation[["id", "fold", "sentiment"]].rename(
                columns={"sentiment": "y_true"}
            ).copy()
            rows["y_pred"] = model.predict(validation.text)
            predictions.append(rows)
    oof = pd.concat(predictions).sort_values("id")
    if not oof.id.is_unique or set(oof.id) != set(dev.id):
        raise ValueError("Expected exactly one held-out prediction per development ID.")
    return oof, cv_report(oof)


def main():
    started = perf_counter()
    config = json.loads(CONFIG.read_text())
    run_id = "baseline_logreg_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "evidence"
    oof, report = run_cv(load_dev(), config)
    pooled = report["pooled"]
    results = {
        "baseline_id": "baseline_logreg", "run_id": run_id, "config": config,
        "primary_metric": {"name": "macro_f1", "value": pooled["macro_f1"],
                           "direction": "higher_is_better", "units": "unitless_0_to_1"},
        **report,
        "runtime_seconds": perf_counter() - started,
        "hardware": "CPU: " + (platform.processor() or platform.machine()),
        "outputs": {"predictions": str(output / "b1_logreg_predictions.csv"),
                    "results": str(output / "b1_logreg_metrics.json")},
    }
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "b1_logreg_predictions.csv", index=False)
    (output / "b1_logreg_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Pooled macro-F1: {pooled['macro_f1']:.6f}; accuracy: {pooled['accuracy']:.6f}")
    print(f"Saved {len(oof)} dev predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
