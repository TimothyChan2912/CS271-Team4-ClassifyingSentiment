"""Regression checks for the YouTube input and variable dataset sizes."""
from contextlib import redirect_stderr
import hashlib
from io import BytesIO, StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.data_cleaning import clean_comments, main, prepare_data
from src.data_io import load_dev, load_test
from src.make_split import build_split


def comments(rows):
    return pd.DataFrame(rows, columns=["CommentID", "CommentText", "Sentiment"])


class DataPipelineTests(unittest.TestCase):
    def test_cleaning_filters_bad_rows_and_preserves_model_text(self):
        raw = comments([
            ("a", "  Great @Alice https://example.com 😀!  ", " Positive "),
            ("b", "great @Bob www.example.org 😀!", "positive"),
            ("c", "Awful comment", "NEGATIVE"),
            ("d", "What time?\nTomorrow", "Neutral"),
            ("e", "Mixed response @a", "Negative"),
            ("f", "mixed response @b", "Positive"),
            ("g", None, "Neutral"),
            ("h", " \n ", "Neutral"),
            ("i", "@user https://example.com", "Neutral"),
            ("j", "Question?", "Mixed"),
            ("k", "No rating", None),
        ])
        raw["VideoTitle"] = "Metadata must not be a model input"
        clean, counts = clean_comments(raw)
        self.assertEqual(clean.columns.tolist(), ["id", "text", "sentiment"])
        self.assertEqual(clean.id.tolist(), ["youtube_a", "youtube_c", "youtube_d"])
        self.assertEqual(clean.text.tolist(), [
            "Great @Alice https://example.com 😀!", "Awful comment", "What time?\nTomorrow"])
        self.assertEqual(clean.sentiment.tolist(), ["positive", "negative", "neutral"])
        self.assertEqual(counts["raw"], 11)
        self.assertEqual(counts["after_empty_removed"], 8)
        self.assertEqual(counts["dropped_invalid_sentiment"], 2)
        self.assertEqual(counts["conflicting_groups"], 1)
        self.assertEqual(counts["conflicting_rows"], 2)
        self.assertEqual(counts["duplicates_removed"], 1)
        self.assertEqual(counts["final"], 3)
        self.assertEqual(counts["posts_with_emoji"], 1)

    def test_csv_limit_counts_records_including_multiline_comments(self):
        raw = comments([
            ("b", 'First line,\nsecond "quoted" line 😀', "Positive"),
            ("a", "Another comment", "Negative"),
            ("c", "A question?", "Neutral"),
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "youtube.csv"
            raw.to_csv(path, index=False)
            subset, counts = prepare_data(path, datasize=2)
            again, _ = prepare_data(path, datasize=2)
            full, full_counts = prepare_data(path, datasize=100)
            pd.testing.assert_frame_equal(subset, again)
            self.assertEqual(counts["raw"], 2)
            self.assertEqual(counts["datasize_requested"], 2)
            self.assertEqual(counts["raw_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(subset.id.tolist(), ["youtube_a", "youtube_b"])
            self.assertEqual(full_counts["raw"], 3)
            self.assertEqual(full_counts["datasize_requested"], 100)
            pd.testing.assert_frame_equal(
                subset, full[full.id.isin(subset.id)].reset_index(drop=True))

    def test_missing_schema_and_unusable_comments_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "required columns"):
            clean_comments(pd.DataFrame({"Text": ["old source"]}))
        with self.assertRaisesRegex(ValueError, "No usable"):
            clean_comments(comments([("a", None, "Positive")]))
        for ids in [["a", "a"], ["", "b"]]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "CommentID"):
                clean_comments(comments([
                    (ids[0], "First comment", "Positive"),
                    (ids[1], "Second comment", "Negative"),
                ]))

    def test_cleaner_rejects_invalid_datasize_without_reading(self):
        for value in ["0", "-5", "1.5", "abc"]:
            with self.subTest(value=value), patch("src.data_cleaning.prepare_data") as prepare, \
                    redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--datasize", value])
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()

    def test_variable_size_split_and_loaders(self):
        for per_class in [10, 30]:
            with self.subTest(per_class=per_class):
                clean, _ = clean_comments(comments([
                    (f"{label}_{i}", f"{label} comment {i}", label)
                    for label in ["negative", "neutral", "positive"] for i in range(per_class)
                ]))
                split, _ = build_split(clean)
                rebuilt, _ = build_split(clean.sample(frac=1, random_state=17))
                pd.testing.assert_frame_equal(split, rebuilt)
                data = clean.to_parquet(index=False)
                assignments = split[["id", "partition", "fold"]].to_csv(index=False)
                with patch("src.data_io.CLEAN", BytesIO(data)), \
                        patch("src.data_io.SPLIT", StringIO(assignments)):
                    dev = load_dev()
                with patch("src.data_io.CLEAN", BytesIO(data)), \
                        patch("src.data_io.SPLIT", StringIO(assignments)), \
                        patch.dict(os.environ, {"FINAL_EVAL": "1"}):
                    test = load_test()
                self.assertEqual(len(dev) + len(test), len(clean))
                self.assertEqual(len(test), per_class * 3 // 5)
                self.assertTrue(set(dev.id).isdisjoint(test.id))
                self.assertEqual(set(dev.id) | set(test.id), set(clean.id))
                for _, fold in dev.groupby("fold"):
                    self.assertEqual(set(fold.sentiment), {"negative", "neutral", "positive"})

    def test_small_sample_requests_larger_datasize(self):
        for labels in [["positive"] * 20, ["negative", "neutral", "positive"],
                       ["negative", "neutral", "positive"] * 4]:
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "--datasize"):
                build_split(pd.DataFrame({"id": range(len(labels)), "sentiment": labels}))

    def test_split_check_after_parquet_and_csv_round_trip(self):
        clean, _ = clean_comments(comments([
            (f"{label}_{i}", f"{label} comment {i}", label)
            for label in ["negative", "neutral", "positive"] for i in range(10)
        ]))
        split, _ = build_split(clean)
        script = Path(__file__).resolve().parents[1] / "src" / "check_split.py"
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / "data" / "processed").mkdir(parents=True)
            (work / "splits").mkdir()
            clean.to_parquet(work / "data" / "processed" / "clean.parquet", index=False)
            split[["id", "partition", "fold"]].to_csv(work / "splits" / "split.csv", index=False)
            result = subprocess.run(
                [sys.executable, str(script)], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("All split checks passed.", result.stdout)

    def test_stale_split_rejected_even_if_row_count_matches(self):
        clean = pd.DataFrame([
            {"id": str(i), "text": f"Comment {i}", "sentiment": "positive"} for i in range(6)
        ])
        split = pd.DataFrame({
            "id": ["0", "1", "2", "3", "4", "old-test-id"],
            "partition": ["dev"] * 5 + ["test"], "fold": [0, 1, 2, 3, 4, -1]})
        with patch("src.data_io.CLEAN", BytesIO(clean.to_parquet(index=False))), \
                patch("src.data_io.SPLIT", StringIO(split.to_csv(index=False))), \
                self.assertRaisesRegex(AssertionError, "different IDs"):
            load_dev()


if __name__ == "__main__":
    unittest.main()
