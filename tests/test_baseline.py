from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data_io import load_dev
from src.metrics import LABELS, binary_pos_neg_accuracy, cv_report, score
from src.baseline_logreg import CONFIG, ROOT, make_model, run_cv


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())
        self.posts = pd.DataFrame([
            {"id": f"{fold}_{label}", "text": f"{label} fold{fold}only", "sentiment": label,
             "partition": "dev", "fold": fold}
            for fold in range(5) for label in LABELS
        ] + [{"id": "protected", "text": None, "sentiment": "DO_NOT_SCORE", "partition": "test", "fold": -1}])

    def load(self, posts=None):
        posts = self.posts if posts is None else posts
        data = BytesIO(posts[["id", "text", "sentiment"]].to_parquet(index=False))
        split = StringIO(self.posts[["id", "partition", "fold"]].to_csv(index=False))
        with patch("src.data_io.CLEAN", data), patch("src.data_io.SPLIT", split):
            return load_dev()

    def test_metric_conventions(self):
        result = score(["negative", "negative", "neutral", "positive"],
                       ["negative", "positive", "neutral", "negative"])
        self.assertEqual(result["confusion_matrix"]["rows_true_cols_pred"],
                         [[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertAlmostEqual(score(["positive"], ["positive"])["macro_f1"], 1 / 3)
        with self.assertRaises(AssertionError):
            score(["positive"], ["unknown"])
        with self.assertRaises(ValueError):
            score(["positive"], [])

    def test_binary_metric_excludes_neutral_and_compares_polarity_scores(self):
        result = binary_pos_neg_accuracy(
            ["positive", "negative", "neutral", "negative", "positive"],
            [[0.10, 0.65, 0.25],  # positive wins even though neutral is largest
             [0.25, 0.65, 0.10],  # negative wins even though neutral is largest
             [0.90, 0.00, 0.10],  # true-neutral row is excluded
             [0.20, 0.60, 0.20],  # polarity tie resolves to negative
             [0.60, 0.10, 0.30]],  # incorrect negative prediction
        )
        self.assertEqual(result, {"n": 4, "accuracy": 3 / 4})

    def test_b1_probabilities_follow_class_names_and_post_ids(self):
        class FixedModel:
            classes_ = ["positive", "negative", "neutral"]

            def fit(self, texts, labels):
                return self

            def predict(self, texts):
                return ["neutral"] * len(texts)

            def predict_proba(self, texts):
                scores = {"negative": [0.10, 0.25, 0.65],
                          "neutral": [0.10, 0.10, 0.80],
                          "positive": [0.25, 0.10, 0.65]}
                return np.array([scores[text.split()[0]] for text in texts])

        dev = self.load().sample(frac=1, random_state=42)
        with patch("src.baseline_logreg.make_model", side_effect=lambda _: FixedModel()):
            predictions, report = run_cv(dev, self.config)
        expected = {"negative": [0.25, 0.65, 0.10],
                    "neutral": [0.10, 0.80, 0.10],
                    "positive": [0.10, 0.65, 0.25]}
        columns = [f"p_{label}" for label in LABELS]
        self.assertEqual(predictions.id.tolist(), sorted(dev.id))
        for label, probabilities in expected.items():
            rows = predictions[predictions.y_true == label]
            np.testing.assert_allclose(rows[columns], [probabilities] * len(rows))
        self.assertTrue(predictions.y_pred.eq("neutral").all())
        self.assertEqual(report["binary_pos_neg_vs_B0"], {"n": 10, "accuracy": 1.0})

    def test_oof_coverage_and_training_only_vocabulary(self):
        dev, models = self.load(), []

        def record_model(config):
            model = make_model(config)
            models.append(model)
            return model

        with patch("src.baseline_logreg.make_model", side_effect=record_model), redirect_stdout(StringIO()):
            predictions, report = run_cv(dev, self.config)
        self.assertNotIn("protected", set(predictions.id))
        self.assertEqual(len(predictions), 15)
        self.assertTrue(predictions.id.is_unique)
        self.assertEqual(set(predictions.id), set(dev.id))
        self.assertEqual(report["pooled"]["n"], 15)
        self.assertEqual(report["binary_pos_neg_vs_B0"]["n"], 10)
        probabilities = predictions[[f"p_{label}" for label in LABELS]].to_numpy()
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(set(predictions.fold), set(range(5)))
        for fold, model in enumerate(models):
            self.assertNotIn(f"fold{fold}only", model[0].vocabulary_)
            self.assertIn(f"fold{(fold + 1) % 5}only", model[0].vocabulary_)

    def test_invalid_ids_are_rejected(self):
        for posts, error in [(pd.concat([self.posts, self.posts.iloc[[0]]]), pd.errors.MergeError),
                             (self.posts.iloc[1:], AssertionError)]:
            with self.assertRaises(error):
                self.load(posts)

    def test_unconverged_fit_is_not_scored(self):
        self.config["classifier"].update(max_iter=1, tol=1e-12)
        with self.assertRaises(ConvergenceWarning), patch("src.baseline_logreg.cv_report") as scoring:
            run_cv(self.load(), self.config)
        scoring.assert_not_called()

    def test_shared_loader_returns_only_dev_rows(self):
        dev = self.load()
        self.assertEqual(len(dev), 15)
        self.assertNotIn("protected", set(dev.id))
        self.assertEqual(set(dev.sentiment), set(LABELS))

    def test_shared_scorer_preserves_recorded_baseline_results(self):
        for baseline in ["b1_logreg", "b2_tweeteval"]:
            with self.subTest(baseline=baseline):
                predictions = pd.read_csv(ROOT / f"evidence/{baseline}_predictions.csv")
                expected = json.loads((ROOT / f"evidence/{baseline}_metrics.json").read_text())
                report = cv_report(predictions)
                self.assertEqual(json.loads(json.dumps(report)), {key: expected[key] for key in report})
                probabilities = predictions[[f"p_{label}" for label in LABELS]].to_numpy()
                self.assertEqual(binary_pos_neg_accuracy(predictions.y_true, probabilities),
                                 expected["binary_pos_neg_vs_B0"])

    def test_run_all_invokes_b1_and_optional_b2(self):
        for options, size in [([], 10000), (["--skip-b2"], 10000),
                              (["--datasize", "2000"], 2000),
                              (["--datasize", "500", "--skip-b2", "--remake-split"], 500)]:
            with self.subTest(options=options), \
                    patch.object(sys, "argv", ["run_all.py", *options]), \
                    patch("subprocess.run", return_value=SimpleNamespace(
                        returncode=0, stdout="", stderr="")) as launch, \
                    patch("builtins.open", mock_open()), redirect_stdout(StringIO()):
                runpy.run_path(str(ROOT / "run_all.py"), run_name="__main__")
            expected = [["src/data_cleaning.py", "--datasize", str(size)],
                        ["src/make_split.py"], ["src/check_split.py"], ["src/baseline_logreg.py"],
                        ["src/run_b1_emoji_ablation.py"]]
            if "--skip-b2" not in options:
                expected.append(["src/baseline_tweeteval.py"])
                expected.append(["src/run_b2_emoji_ablation.py"])
            expected.append(["src/run_summary.py", *(["--skip-b2"] if "--skip-b2" in options else [])])
            self.assertEqual([call.args[0][1:] for call in launch.call_args_list], expected)

    def test_run_all_full_report_includes_proposed_and_guarded_test_results(self):
        with patch.object(sys, "argv", ["run_all.py", "--proposed", "--final"]), \
                patch("subprocess.run", return_value=SimpleNamespace(
                    returncode=0, stdout="", stderr="")) as launch, \
                patch("builtins.open", mock_open()), redirect_stdout(StringIO()):
            runpy.run_path(str(ROOT / "run_all.py"), run_name="__main__")
        commands = [call.args[0][1:] for call in launch.call_args_list]
        self.assertIn(["src/proposed_finetune.py", "cv"], commands)
        self.assertIn(["src/proposed_finetune.py", "ablation"], commands)
        self.assertIn(["src/proposed_finetune.py", "final"], commands)
        summary = next(call for call in launch.call_args_list if "src/run_summary.py" in call.args[0])
        self.assertEqual(summary.args[0][1:], ["src/run_summary.py", "--proposed", "--final"])
        self.assertEqual(summary.kwargs["env"]["FINAL_EVAL"], "1")
        self.assertLess(commands.index(["src/final_eval.py"]), commands.index(summary.args[0][1:]))

    def test_run_all_rejects_invalid_datasize_before_starting_work(self):
        from contextlib import redirect_stderr

        for value in ["0", "-10", "1.5", "abc"]:
            with self.subTest(value=value), \
                    patch.object(sys, "argv", ["run_all.py", "--datasize", value]), \
                    patch("subprocess.run") as launch, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runpy.run_path(str(ROOT / "run_all.py"), run_name="__main__")
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
