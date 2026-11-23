
import json
import platform
from pathlib import Path
from time import perf_counter

from baseline_logreg import run_cv
from data_io import load_dev


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baseline_logreg.json"
OUTPUT = ROOT / "evidence" / "ablations" / "b1"


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    dev = load_dev()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    conditions = {
        "described": "emoji-described",
        "removed": "emoji-removed",
    }

    for mode, condition_name in conditions.items():

        method_id = f"B1-{condition_name.upper()}"

        print(f"\nRunning {method_id}...")

        started = perf_counter()

        predictions, report = run_cv(
            dev,
            config,
            emoji_mode=mode,
        )

        runtime = perf_counter() - started

        filename_prefix = condition_name

        prediction_file = (
            OUTPUT / f"{filename_prefix}_predictions.csv"
        )
        metrics_file = (
            OUTPUT / f"{filename_prefix}_metrics.json"
        )

        predictions.to_csv(prediction_file, index=False)

        results = {
            "method_id": method_id,
            "model": "TF-IDF + Logistic Regression",
            "emoji_mode": mode,
            "model_card_preprocessing": False,
            "partition": "dev",
            "cross_validation": "5-fold stratified",
            "test_exposure": "none",
            "config": config,
            "primary_metric": {
                "name": "macro_f1",
                "value": report["pooled"]["macro_f1"],
                "direction": "higher_is_better",
                "units": "unitless_0_to_1",
            },
            **report,
            "runtime_seconds": runtime,
            "hardware": "CPU: " + (
                platform.processor() or platform.machine()
            ),
            "outputs": {
                "predictions": str(prediction_file),
                "metrics": str(metrics_file),
            },
        }

        metrics_file.write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        print(
            f"Macro-F1: {report['pooled']['macro_f1']:.6f}"
        )
        print(
            f"Per-fold Macro-F1: "
            f"{report['per_fold_macro_f1_mean']:.6f} "
            f"± {report['per_fold_macro_f1_std']:.6f}"
        )
        print(f"Saved predictions: {prediction_file}")
        print(f"Saved metrics: {metrics_file}")


if __name__ == "__main__":
    main()
