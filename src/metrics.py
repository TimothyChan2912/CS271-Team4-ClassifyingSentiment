"""Shared evaluation metrics (PROTOCOL.md Section 5).

Every baseline and the fine-tuned model must score through these functions so
that label order, zero-division handling, and the pooled-vs-per-fold rule are
identical across methods.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]  # fixed order, PROTOCOL.md Section 5


def score(y_true, y_pred):
    """Macro-F1 (primary), accuracy, per-class F1, and confusion matrix."""
    y_true, y_pred = list(y_true), list(y_pred)
    bad = set(y_true + y_pred) - set(LABELS)
    assert not bad, f"unknown labels: {bad}"
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {lab: float(v) for lab, v in zip(LABELS, per_class)},
        "confusion_matrix": {
            "labels": LABELS,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def binary_pos_neg_accuracy(y_true, scores):
    """Positive-vs-negative accuracy, for the B0 (Go et al., 2009) reference only.

    Keeps posts whose true label is negative or positive, then predicts the
    larger of the negative and positive scores (neutral is ignored).
    `scores` is an (n, 3) array in LABELS order (probabilities or logits).
    Not directly comparable to the 83% figure: different data and test set.
    """
    y_true = np.asarray(list(y_true))
    scores = np.asarray(scores, dtype=float)
    assert scores.shape == (len(y_true), 3), "scores must be (n, 3) in LABELS order"
    keep = y_true != "neutral"
    pred = np.where(scores[keep, 2] > scores[keep, 0], "positive", "negative")
    return {"n": int(keep.sum()), "accuracy": float((pred == y_true[keep]).mean())}


def cv_report(preds: pd.DataFrame):
    """Dev-partition report from out-of-fold predictions.

    `preds` needs columns `fold`, `y_true`, `y_pred`, with exactly one row per
    dev post. Pooled macro-F1 is the headline number; per-fold mean and
    standard deviation (sample std, ddof=1) are reported alongside it.
    """
    assert set(preds["fold"]) == {0, 1, 2, 3, 4}, "need all 5 folds, never a selected fold"
    per_fold = {int(k): score(g["y_true"], g["y_pred"]) for k, g in preds.groupby("fold")}
    fold_f1 = np.array([per_fold[k]["macro_f1"] for k in sorted(per_fold)])
    return {
        "pooled": score(preds["y_true"], preds["y_pred"]),
        "per_fold_macro_f1": {k: per_fold[k]["macro_f1"] for k in sorted(per_fold)},
        "per_fold_macro_f1_mean": float(fold_f1.mean()),
        "per_fold_macro_f1_std": float(fold_f1.std(ddof=1)),
    }
