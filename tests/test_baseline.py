from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

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
        with patch("src.data_io.CLEAN", data), patch("src.data_io.SPLIT", split), \
                patch("src.data_io.EXPECTED_DEV", 15):
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

    def test_shared_scorer_preserves_recorded_b2_results(self):
        predictions = pd.read_csv(ROOT / "evidence/b2_tweeteval_predictions.csv")
        expected = json.loads((ROOT / "evidence/b2_tweeteval_metrics.json").read_text())
        report = cv_report(predictions)
        self.assertEqual(json.loads(json.dumps(report)), {key: expected[key] for key in report})
        probabilities = predictions[[f"p_{label}" for label in LABELS]].to_numpy()
        self.assertEqual(binary_pos_neg_accuracy(predictions.y_true, probabilities),
                         expected["binary_pos_neg_vs_B0"])

    def test_run_all_invokes_b1_and_optional_b2(self):
        for options in [[], ["--skip-b2"]]:
            with self.subTest(options=options), \
                    patch.object(sys, "argv", ["run_all.py", *options]), \
                    patch("subprocess.run", return_value=SimpleNamespace(
                        returncode=0, stdout="", stderr="")) as launch, \
                    patch("builtins.open", mock_open()), redirect_stdout(StringIO()):
                runpy.run_path(str(ROOT / "run_all.py"), run_name="__main__")
            expected = [["src/data_cleaning.py"], ["src/check_split.py"], ["src/baseline_logreg.py"]]
            if not options:
                expected.append(["src/baseline_tweeteval.py"])
            self.assertEqual([call.args[0][1:] for call in launch.call_args_list], expected)


if __name__ == "__main__":
    unittest.main()
