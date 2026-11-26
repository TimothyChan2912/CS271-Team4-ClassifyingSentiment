from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.final_eval import aggregate, run_b1, run_b2
from src.metrics import LABELS, align_predictions
from src.run_summary import build_report, main, prediction_paths, summarize_predictions, write_report


class RunSummaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.evidence = Path(self.folder.name)
        self.dev = pd.DataFrame([
            {"id": f"dev_{i}", "text": f"Comment {i}" + (" 😀" if i < 2 else ""),
             "sentiment": ["positive", "negative", "neutral"][i % 3], "fold": i // 3}
            for i in range(15)])
        self.test = pd.DataFrame([
            {"id": f"test_{i}", "text": f"Test {i}" + (" 😀" if i < 2 else ""), "sentiment": label}
            for i, label in enumerate(["positive", "negative", "neutral", "positive"])])
        self.split_summary = {
            "partition_sizes": {"dev": 15, "test": 4},
            "class_counts_by_partition": {
                "dev": {"positive": 5, "negative": 5, "neutral": 5},
                "test": {"positive": 2, "negative": 1, "neutral": 1}},
            "emoji_counts_by_partition": {"dev": 2, "test": 2}}
        self.selection = (2e-5, [42, 43])
        grid = {"selected_learning_rate": 2e-5, "grid": {"2e-05": {"seeds": [42, 43]}}}
        (self.evidence / "proposed").mkdir()
        (self.evidence / "proposed" / "cv_grid.json").write_text(json.dumps(grid))

    def save_predictions(self, path, posts, predicted=None):
        frame = posts[["id", "sentiment"]].rename(columns={"sentiment": "y_true"}).copy()
        frame["fold"] = posts["fold"] if "fold" in posts else -1
        frame["y_pred"] = predicted if predicted is not None else posts.sentiment.tolist()
        for label in LABELS:
            frame[f"p_{label}"] = frame["y_pred"].eq(label).astype(float)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sample(frac=1, random_state=7).to_csv(path, index=False)

    def populate(self):
        for model in ["B1", "B2", "P"]:
            for mode in ["raw", "removed", "described"]:
                for partition, posts in [("dev", self.dev), ("test", self.test)]:
                    paths = prediction_paths(self.evidence, model, mode, partition, self.selection)
                    for index, path in enumerate(paths):
                        predicted = posts.sentiment.tolist()
                        if partition == "test":
                            predicted = (["negative", "negative", "neutral", "positive"] if index == 0
                                         else ["positive", "positive", "neutral", "positive"])
                        self.save_predictions(path, posts, predicted)

    def report(self, **kwargs):
        options = {"include_proposed": True}
        options.update(kwargs)
        return build_report(self.dev, self.test, self.split_summary, self.evidence, **options)

    def test_report_covers_all_models_modes_counts_and_emoji_subset(self):
        self.populate()
        report = self.report()
        self.assertEqual(len(report["results"]), 9)
        self.assertEqual(report["class_counts"]["test"],
                         {"negative": 1, "neutral": 1, "positive": 2, "total": 4, "emoji_posts": 2})
        for row in report["results"]:
            self.assertEqual(row["dev_positive"], 5)
            self.assertEqual(row["dev_negative"], 5)
            self.assertEqual(row["dev_macro_f1"], 1.0)
            self.assertEqual(row["dev_accuracy"], 1.0)
            self.assertEqual(row["test_total"], 4)
            self.assertEqual(row["test_accuracy"], 0.75)
            # The original emoji subset remains the same even for the removed variant.
            self.assertEqual(row["test_emoji_posts"], 2)
            self.assertEqual(row["test_emoji_accuracy"], 0.5)
            self.assertEqual(row["dev_status"], "complete")
            self.assertEqual(row["test_status"], "complete")
            self.assertEqual(row["test_runs"], 2 if row["model"] == "P" else 1)
            self.assertAlmostEqual(row["test_macro_f1"], 31 / 45 if row["model"] == "P" else 7 / 9)

    def test_only_selected_learning_rate_and_seeds_are_used(self):
        self.populate()
        unrelated = self.evidence / "proposed" / "cv" / "P_raw_lr1e-05_seed99_predictions.csv"
        unrelated.write_text("invalid predictions should never be read")
        report = self.report()
        self.assertEqual(report["selected_learning_rate"], 2e-5)
        self.assertEqual(report["proposed_seeds"], [42, 43])
        self.assertEqual(next(row for row in report["results"] if row["model"] == "P")["dev_runs"], 2)

    def test_no_emojis_produces_null_accuracy(self):
        self.populate()
        posts = self.test.copy()
        posts["text"] = "No emoji here"
        path = prediction_paths(self.evidence, "B1", "removed", "test")
        result = summarize_predictions(path, posts)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertIsNone(result["emoji_accuracy"])

    def test_missing_results_are_not_reported_as_zero(self):
        report = self.report()
        for row in report["results"]:
            self.assertEqual(row["dev_status"], "not_run")
            self.assertIsNone(row["dev_macro_f1"])
            self.assertIsNone(row["test_accuracy"])

    def test_partial_seeds_are_rejected(self):
        paths = prediction_paths(self.evidence, "P", "raw", "test", self.selection)
        self.save_predictions(paths[0], self.test)
        with self.assertRaisesRegex(ValueError, "Incomplete seed results"):
            summarize_predictions(paths, self.test)

    def test_stale_labels_folds_and_ids_are_rejected(self):
        path = self.evidence / "predictions.csv"
        self.save_predictions(path, self.dev)
        original = pd.read_csv(path)
        for column, value in [("id", "old-id"), ("y_true", "old-label"), ("fold", -1)]:
            wrong = original.copy()
            wrong.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                align_predictions(wrong, self.dev)
        duplicate = pd.concat([original.iloc[1:], original.iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError):
            align_predictions(duplicate, self.dev)

    def test_skipped_models_and_test_results_are_not_read(self):
        self.populate()
        path = prediction_paths(self.evidence, "B2", "raw", "dev")[0]
        path.write_text("stale B2 file")
        for path in (self.evidence / "final").glob("*.csv"):
            path.write_text("test files must not be read")
        report = build_report(self.dev, None, self.split_summary, self.evidence,
                              include_proposed=False, include_b2=False)
        for row in report["results"]:
            self.assertEqual(row["test_status"], "not_run")
            self.assertEqual(row["test_positive"], 2)
            if row["model"] != "B1":
                self.assertEqual(row["dev_status"], "not_run")

    def test_csv_and_json_contain_requested_columns(self):
        self.populate()
        report = self.report()
        csv_path, json_path = write_report(report, self.evidence)
        saved = pd.read_csv(csv_path)
        self.assertEqual(len(saved), 9)
        self.assertTrue({"dev_positive", "dev_negative", "test_positive", "test_negative",
                         "dev_macro_f1", "test_macro_f1", "dev_accuracy", "test_accuracy",
                         "dev_emoji_accuracy", "test_emoji_accuracy"}.issubset(saved.columns))
        self.assertEqual(json.loads(json_path.read_text()), report)

    def test_default_report_does_not_open_test_partition(self):
        (self.evidence / "split_summary.json").write_text(json.dumps(self.split_summary))
        (self.evidence / "prepare_data_counts.json").write_text('{"raw": 19}')
        with patch("src.run_summary.EVIDENCE", self.evidence), \
                patch("src.run_summary.load_dev", return_value=self.dev), \
                patch("src.run_summary.load_test") as load_test, redirect_stdout(StringIO()):
            main(["--skip-b2"])
        load_test.assert_not_called()

    def test_final_b1_applies_ablation_to_both_train_and_test(self):
        model = Mock()
        model.fit.return_value = model
        model.classes_ = LABELS
        model.predict_proba.return_value = np.tile([0.1, 0.2, 0.7], (4, 1))
        for mode, marker in [("raw", "😀"), ("removed", None), ("described", "grinning face")]:
            with self.subTest(mode=mode), \
                    patch("src.baseline_logreg.make_model", return_value=model), \
                    patch("src.final_eval.save") as save:
                run_b1(self.dev, self.test, mode)
                trained = model.fit.call_args.args[0].tolist()
                evaluated = model.predict_proba.call_args.args[0].tolist()
                for texts in [trained, evaluated]:
                    self.assertEqual("😀" in texts[0], mode == "raw")
                    if marker:
                        self.assertIn(marker, texts[0])
                self.assertIn("😀", save.call_args.args[1].text.iloc[0])
                self.assertEqual(save.call_args.args[3]["emoji_mode"], mode)

    def test_final_b2_applies_ablation_and_uses_baseline_revision(self):
        revision = "a" * 40
        metadata = json.dumps({"config": {"model_commit": revision}})
        predict = Mock(return_value=(np.tile([0.1, 0.2, 0.7], (4, 1)), {}))
        with patch("src.baseline_tweeteval.predict", predict), \
                patch("src.final_eval.Path.read_text", return_value=metadata), \
                patch("src.final_eval.save"):
            run_b2(self.test, "cpu", "described")
        self.assertIn("grinning face", predict.call_args.args[0][0])
        self.assertEqual(predict.call_args.args[-1], revision)

    def test_final_aggregation_includes_baseline_ablations_and_excludes_skipped_b2(self):
        self.populate()
        for path in (self.evidence / "final").glob("B2*"):
            path.write_text("stale B2 file")
        with patch("src.final_eval.FINAL", self.evidence / "final"), \
                patch("src.final_eval.N_BOOT", 5), redirect_stdout(StringIO()):
            table, _ = aggregate(methods=["B1", "B1-removed", "B1-described"], test=self.test)
        self.assertEqual(set(table.method), {"B1", "B1-removed", "B1-described"})

    def test_figure_reporting_skips_b2_and_uses_current_class_counts(self):
        from src import make_figures

        (self.evidence / "b1_logreg_metrics.json").write_text(json.dumps({
            "pooled": {"per_class_f1": {label: 0.5 for label in LABELS}}}))
        (self.evidence / "b2_tweeteval_metrics.json").write_text("stale B2 metrics")
        (self.evidence / "split_summary.json").write_text(json.dumps(self.split_summary))
        with patch("src.make_figures.EVID", self.evidence):
            self.assertEqual(set(make_figures.dev_reports(skip_b2=True)), {"B1"})
            figure = make_figures.fig2(skip_b2=True)
        self.assertEqual([tick.get_text() for tick in figure.axes[0].get_xticklabels()],
                         ["Negative\n(n=5)", "Neutral\n(n=5)", "Positive\n(n=5)"])
        make_figures.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
