"""Count proposed-model fits and carry progress across runner subprocesses."""
import os


def training_run_counts(cfg, *, lrs=None, seeds=None, modes=None):
    n_seeds = len(seeds or cfg["seeds"])
    n_modes = len(modes or cfg["ablation_emoji_modes"])
    return {
        "cv": len(lrs or cfg["learning_rates"]) * n_seeds * cfg["n_folds"],
        "ablation": n_modes * n_seeds * cfg["n_folds"],
        "final": (1 + n_modes) * n_seeds,
    }


class TrainingProgress:
    def __init__(self, total):
        self.current = int(os.environ.get("PROPOSED_TRAINING_OFFSET", "0"))
        self.total = int(os.environ.get("PROPOSED_TRAINING_TOTAL", str(total)))

    def start(self, description):
        self.current += 1
        print(f"Training run {self.current}/{self.total}: {description}", flush=True)
