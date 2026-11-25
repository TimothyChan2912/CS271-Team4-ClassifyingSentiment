"""Checks for the proposed-method pipeline that run without downloading a model.

    python -m unittest tests.test_proposed
"""
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data_io
from src.metrics import LABELS
from src.proposed_finetune import balanced_weights, run_cv, select


def fake_dev():
    rows = [{"id": f"p{fold}_{lab}_{k}", "fold": fold, "text": f"{lab} post {fold}-{k} 😀",
             "sentiment": lab} for fold in range(5) for lab in LABELS for k in range(2)]
    return pd.DataFrame(rows)


class ProposedTests(unittest.TestCase):
    def test_test_partition_is_locked_without_final_eval(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                data_io.load_test()

    def test_class_weights_use_training_labels_only(self):
        w = balanced_weights(["negative"] * 6 + ["neutral"] * 2 + ["positive"] * 4)
        np.testing.assert_allclose(w, [12 / 18, 12 / 6, 12 / 12])

    def test_cv_trains_without_held_out_fold_and_predicts_each_post_once(self):
        dev, seen = fake_dev(), []

        def fake_train(texts, labels, cfg, lr, seed, device):
            seen.append(set(texts))
            return None, None, [{"epoch": 1, "train_loss": 0.0}]

        def fake_predict(tok, model, texts, cfg, device):
            return np.array([[0.1, 0.1, 0.8] if t.startswith("positive") else
                             [0.1, 0.8, 0.1] if t.startswith("neutral") else
                             [0.8, 0.1, 0.1] for t in texts])

        oof, report, _ = run_cv(dev, {"n_folds": 5}, 2e-5, 42, "removed", "cpu",
                                train_fn=fake_train, predict_fn=fake_predict)
        self.assertEqual(len(oof), len(dev))
        self.assertTrue(oof.id.is_unique)
        self.assertEqual(report["pooled"]["macro_f1"], 1.0)
        for fold, texts in enumerate(seen):
            held = set(dev[dev.fold == fold].text.str.replace(" 😀", ""))
            self.assertTrue(held.isdisjoint(texts), "held-out fold leaked into training")
            self.assertFalse(any("😀" in t for t in texts), "emoji_mode=removed not applied")

    def test_emoji_conditions_only_change_posts_with_emoji(self):
        from src.emoji_processing import convert_emojis_to_text, remove_emojis
        plain = "Time: 5pm at the_park  today\nsee you"
        self.assertEqual(convert_emojis_to_text(plain), plain)
        self.assertEqual(remove_emojis(plain), plain)
        self.assertEqual(convert_emojis_to_text("Great day 😀!"), "Great day grinning face !")
        self.assertEqual(remove_emojis("Great day 😀!"), "Great day !")

    def test_selection_rule_breaks_ties_by_lower_fold_sd(self):
        grid = {"1e-05": {"mean_cv_macro_f1": 0.700, "mean_fold_sd": 0.03},
                "2e-05": {"mean_cv_macro_f1": 0.703, "mean_fold_sd": 0.06},
                "3e-05": {"mean_cv_macro_f1": 0.650, "mean_fold_sd": 0.01}}
        self.assertEqual(select(grid, 0.005), "1e-05")
        self.assertEqual(select(grid, 0.0), "2e-05")


if __name__ == "__main__":
    unittest.main()
