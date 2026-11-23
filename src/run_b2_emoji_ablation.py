
import json
import platform
from pathlib import Path
from time import perf_counter

from baseline_tweeteval import (
    ID2SENTIMENT,
    MAX_LENGTH,
    MODEL,
    predict,
)
from data_io import load_dev
from metrics import LABELS, binary_pos_neg_accuracy, cv_report
from emoji_processing import (
    convert_emojis_to_text,
    remove_emojis,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "ablations" / "b2"

BATCH_SIZE = 32
REVISION = "main"


def main():
    dev = load_dev()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    conditions = {
        "described": convert_emojis_to_text,
        "removed": remove_emojis,
    }

    for mode, transform in conditions.items():

        method_id = f"B2-EMOJI-{mode.upper()}"

        print(f"\nRunning {method_id}...")

        started = perf_counter()

        # Apply the selected emoji transformation.
        texts = dev["text"].apply(transform).tolist()

        # Reuse the original B2 inference implementation.
        probs, info = predict(
            texts,
            batch_size=BATCH_SIZE,
            device_arg="auto",
            revision=REVISION,
        )

        runtime = perf_counter() - started

        assert probs.shape == (len(dev), 3)

        results_df = dev.copy()
        results_df["y_true"] = results_df["sentiment"]
        results_df["y_pred"] = [
            ID2SENTIMENT[i] for i in probs.argmax(axis=1)
        ]

        for j, label in enumerate(LABELS):
            results_df[f"p_{label}"] = probs[:, j]

        report = cv_report(
            results_df[["fold", "y_true", "y_pred"]]
        )

        report["binary_pos_neg_vs_B0"] = binary_pos_neg_accuracy(
            results_df["y_true"], probs
        )

        report["config"] = {
            "id": method_id,
            "model": MODEL,
            "revision_requested": REVISION,
            "max_length": MAX_LENGTH,
            "truncation": True,
            "zero_shot": True,
            "batch_size": BATCH_SIZE,
            "emoji_mode": mode,
            "model_card_preprocessing": False,
            "partition": "dev",
            "cross_validation": "5-fold reporting",
            "test_exposure": "none",
            **info,
        }

        filename_prefix = f"emoji-{mode}"

        prediction_file = (
            OUTPUT / f"{filename_prefix}_predictions.csv"
        )
        metrics_file = (
            OUTPUT / f"{filename_prefix}_metrics.json"
        )

        columns = (
            ["id", "fold", "y_true", "y_pred"]
            + [f"p_{label}" for label in LABELS]
        )

        results_df[columns].to_csv(
            prediction_file,
            index=False,
        )

        report["runtime_seconds"] = runtime
        report["hardware"] = "CPU: " + (
            platform.processor() or platform.machine()
        )
        report["outputs"] = {
            "predictions": str(prediction_file),
            "metrics": str(metrics_file),
        }

        metrics_file.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        pooled = report["pooled"]

        print(
            f"Macro-F1: {pooled['macro_f1']:.6f} | "
            f"Accuracy: {pooled['accuracy']:.6f}"
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
