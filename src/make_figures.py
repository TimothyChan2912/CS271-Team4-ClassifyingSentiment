"""Paper figures, built only from saved evidence files.

    python src/make_figures.py
Writes paper/figures/fig1_test_macro_f1.{pdf,png} and fig2_dev_per_class_f1.{pdf,png}.

Fig. 1: test macro-F1 per method with 95% bootstrap intervals (evidence/final/test_results.csv).
Fig. 2: development per-class F1 (out-of-fold) for each method; P bars are means over seeds
        with SD error bars. It shows where the methods differ (neutral) and what the
        emoji ablation changes.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
FIGS = ROOT / "paper" / "figures"
LABELS = ["negative", "neutral", "positive"]
# Fixed categorical order (validated: CVD-safe adjacent pairs); identity is also carried by
# axis labels / legend text, never by color alone.
COLORS = {"B1": "#2a78d6", "B2": "#eb6834", "P-raw": "#1baf7a",
          "P-removed": "#eda100", "P-described": "#e87ba4"}
NAMES = {"B1": "B1 TF-IDF+LR", "B2": "B2 zero-shot", "P-raw": "P fine-tuned",
         "P-removed": "P, emojis removed", "P-described": "P, emojis as text"}
INK, MUTED = "#2b2b2a", "#6b6a66"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def fig1():
    table = pd.read_csv(EVID / "final" / "test_results.csv")
    methods = [m for m in COLORS if m in set(table.method)]
    t = table.set_index("method").loc[methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(x, t.macro_f1, width=0.62, color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1)
    err = np.vstack([t.macro_f1 - t.ci95_low, t.ci95_high - t.macro_f1])
    ax.errorbar(x, t.macro_f1, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for xi, v, hi in zip(x, t.macro_f1, t.ci95_high):
        ax.text(xi, hi + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [NAMES[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Test macro-F1 (0 to 1)", fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    style(ax)
    fig.tight_layout()
    return fig


def dev_reports(skip_b2=False):
    """{method: [pooled dev reports]} from saved out-of-fold predictions/metrics."""
    out = {}
    for method, name in (("B1", "b1_logreg"), ("B2", "b2_tweeteval")):
        path = EVID / f"{name}_metrics.json"
        if path.exists() and not (skip_b2 and method == "B2"):
            out[method] = [json.loads(path.read_text())]
    grid_path = EVID / "proposed" / "cv_grid.json"
    if not grid_path.exists():
        return out
    grid = json.loads(grid_path.read_text())
    lr = grid["selected_learning_rate"]
    for mode in ("raw", "removed", "described"):
        folder = "cv" if mode == "raw" else "ablation"
        files = sorted((EVID / "proposed" / folder).glob(f"P_{mode}_lr{lr:g}_seed*_metrics.json"))
        if files:
            out[f"P-{mode}"] = [json.loads(p.read_text()) for p in files]
    return out


def fig2(skip_b2=False):
    reports = dev_reports(skip_b2=skip_b2)
    methods = [m for m in COLORS if m in reports]
    width = 0.8 / len(methods)
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for k, m in enumerate(methods):
        vals = np.array([[r["pooled"]["per_class_f1"][lab] for lab in LABELS] for r in reports[m]])
        mean = vals.mean(axis=0)
        sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else None
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, mean, width=width * 0.9, color=COLORS[m], edgecolor="white", linewidth=0.8,
               label=NAMES[m])
        if sd is not None:
            ax.errorbar(pos, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    counts = json.loads((EVID / "split_summary.json").read_text())["class_counts_by_partition"]["dev"]
    ax.set_xticks(x, [f"{label.title()}\n(n={counts.get(label, 0)})" for label in LABELS])
    ax.set_ylabel("Dev F1, out-of-fold (0 to 1)", fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    style(ax)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-b2", action="store_true")
    args = parser.parse_args(argv)
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, make in [("fig1_test_macro_f1", fig1),
                       ("fig2_dev_per_class_f1", lambda: fig2(skip_b2=args.skip_b2))]:
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote paper/figures/{name}.pdf")


if __name__ == "__main__":
    main()
