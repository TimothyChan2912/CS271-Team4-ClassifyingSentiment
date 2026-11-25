"""Proposed method (P): RoBERTa fine-tuned on our sentiment labels.

The model starts from the same pretrained checkpoint that B2 uses zero-shot
(cardiffnlp/twitter-roberta-base-sentiment at the commit recorded for B2), so
B2 is exactly P without fine-tuning. Fine-tuning updates every weight with a
class-weighted cross-entropy loss; class weights are computed on the training
rows only. Configuration: configs/proposed_finetune.json.

Commands (run from the repo root):
    python src/proposed_finetune.py cv          # learning-rate grid on dev, 5 folds x 3 seeds
    python src/proposed_finetune.py ablation    # emoji ablation on dev at the selected setting
    FINAL_EVAL=1 python src/proposed_finetune.py final
        # train on all dev with the selected setting (3 seeds, each emoji mode),
        # predict the test partition once, save the seed-42 checkpoint
    FINAL_EVAL=1 python src/proposed_finetune.py eval-checkpoint --path checkpoints/proposed_raw_seed42
        # score saved parameters on test without retraining

Options: --seeds 42 --lrs 2e-5 to run a subset; --device cpu|mps|cuda.
"""
import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

if __package__:
    from .data_io import load_dev, load_test
    from .emoji_processing import convert_emojis_to_text, remove_emojis
    from .metrics import LABELS, binary_pos_neg_accuracy, cv_report, score
else:
    from data_io import load_dev, load_test
    from emoji_processing import convert_emojis_to_text, remove_emojis
    from metrics import LABELS, binary_pos_neg_accuracy, cv_report, score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "proposed_finetune.json"
OUT = ROOT / "evidence" / "proposed"
FINAL_OUT = ROOT / "evidence" / "final"
CKPT = ROOT / "checkpoints"
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
EMOJI = {"raw": lambda t: t, "removed": remove_emojis, "described": convert_emojis_to_text}


# ---------------------------------------------------------------- utilities
def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested="auto"):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware(device):
    import subprocess
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        except Exception:
            pass
    return f"{platform.system()} {platform.release()}, {cpu}, device={device}"


def versions():
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__}


def balanced_weights(labels):
    """sklearn-style 'balanced' weights n / (k * count_c), from training labels only."""
    counts = np.array([sum(lab == c for lab in labels) for c in LABELS], dtype=float)
    return len(labels) / (len(LABELS) * np.maximum(counts, 1.0))


def load_pretrained(cfg):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kw = {"revision": cfg["revision"]} if cfg.get("revision") else {}
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], **kw)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS),
        id2label={i: lab for i, lab in enumerate(LABELS)}, label2id=LABEL2ID, **kw)
    return tok, model


# ------------------------------------------------------------ train / predict
def train_model(texts, labels, cfg, lr, seed, device):
    """Fine-tune a fresh copy of the pretrained checkpoint. Returns tok, model, log."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    tok, model = load_pretrained(cfg)
    model.to(device).train()
    enc = tok(list(texts), truncation=True, max_length=cfg["max_length"])
    y = [LABEL2ID[lab] for lab in labels]
    weight = None
    if cfg.get("class_weight") == "balanced":
        weight = torch.tensor(balanced_weights(list(labels)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    n, bs = len(y), cfg["batch_size"]
    steps = cfg["epochs"] * ((n + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    gen = torch.Generator().manual_seed(seed)
    log = []
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(n, generator=gen).tolist()
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            batch = tok.pad({k: [enc[k][i] for i in idx] for k in enc}, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            target = torch.tensor([y[i] for i in idx], device=device)
            loss = loss_fn(model(**batch).logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        log.append({"epoch": epoch + 1, "train_loss": round(total / n, 5)})
    model.eval()
    return tok, model, log


def predict_probs(tok, model, texts, cfg, device):
    import torch
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch = tok(list(texts[start:start + 32]), truncation=True,
                        max_length=cfg["max_length"], padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def prediction_frame(frame, probs, fold=None):
    out = frame[["id"]].copy()
    out["fold"] = frame["fold"].values if fold is None else fold
    out["y_true"] = frame["sentiment"].values
    out["y_pred"] = [LABELS[i] for i in probs.argmax(axis=1)]
    for j, lab in enumerate(LABELS):
        out[f"p_{lab}"] = probs[:, j]
    return out


# ------------------------------------------------------------------- CV run
def run_cv(dev, cfg, lr, seed, emoji_mode, device, train_fn=train_model, predict_fn=predict_probs):
    """Out-of-fold predictions over all dev folds for one (lr, seed, emoji mode)."""
    dev = dev.copy()
    dev["text"] = dev["text"].map(EMOJI[emoji_mode])
    parts, logs = [], {}
    for fold in range(cfg["n_folds"]):
        train, held = dev[dev.fold != fold], dev[dev.fold == fold]
        tok, model, log = train_fn(train.text.tolist(), train.sentiment.tolist(), cfg, lr, seed, device)
        parts.append(prediction_frame(held, predict_fn(tok, model, held.text.tolist(), cfg, device)))
        logs[fold] = log
        del model
    oof = pd.concat(parts).sort_values("id").reset_index(drop=True)
    if not oof.id.is_unique or set(oof.id) != set(dev.id):
        raise ValueError("Expected exactly one held-out prediction per development ID.")
    report = cv_report(oof)
    report["binary_pos_neg_vs_B0"] = binary_pos_neg_accuracy(
        oof.y_true, oof[[f"p_{lab}" for lab in LABELS]].to_numpy())
    return oof, report, logs


def run_and_save(dev, cfg, lr, seed, emoji_mode, device, subdir, **fns):
    started = perf_counter()
    oof, report, logs = run_cv(dev, cfg, lr, seed, emoji_mode, device, **fns)
    runtime = perf_counter() - started
    name = f"P_{emoji_mode}_lr{lr:g}_seed{seed}"
    folder = OUT / subdir
    folder.mkdir(parents=True, exist_ok=True)
    oof.to_csv(folder / f"{name}_predictions.csv", index=False)
    result = {"method_id": f"P-{emoji_mode}", "run_id": name, "partition": "dev",
              "test_exposure": "none", "learning_rate": lr, "seed": seed,
              "emoji_mode": emoji_mode, "config": cfg, **report,
              "train_log": logs, "runtime_seconds": round(runtime, 1),
              "hardware": hardware(device), "versions": versions()}
    (folder / f"{name}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: pooled macro-F1 {report['pooled']['macro_f1']:.4f} | "
          f"fold mean {report['per_fold_macro_f1_mean']:.4f} +/- "
          f"{report['per_fold_macro_f1_std']:.4f} | {runtime:.0f}s", flush=True)
    return result


def summarize(results):
    """Average across seeds for one setting. Selection uses the mean per-fold macro-F1."""
    fold_means = [r["per_fold_macro_f1_mean"] for r in results]
    pooled = [r["pooled"]["macro_f1"] for r in results]
    fold_sds = [r["per_fold_macro_f1_std"] for r in results]
    return {"seeds": [r["seed"] for r in results],
            "mean_cv_macro_f1": float(np.mean(fold_means)),
            "sd_across_seeds": float(np.std(fold_means, ddof=1)) if len(results) > 1 else 0.0,
            "mean_fold_sd": float(np.mean(fold_sds)),
            "pooled_macro_f1_mean": float(np.mean(pooled)),
            "per_class_f1_mean": {lab: float(np.mean([r["pooled"]["per_class_f1"][lab] for r in results]))
                                  for lab in LABELS}}


def select(grid, tolerance):
    """Highest mean CV macro-F1; within `tolerance` of the best counts as a tie,
    broken by lower mean fold SD (PROTOCOL.md tie rule: simpler, then lower variance)."""
    best = max(s["mean_cv_macro_f1"] for s in grid.values())
    tied = {lr: s for lr, s in grid.items() if best - s["mean_cv_macro_f1"] <= tolerance}
    return min(tied, key=lambda lr: (tied[lr]["mean_fold_sd"], float(lr)))


# ----------------------------------------------------------------- commands
def cmd_cv(cfg, args, device, **fns):
    dev = load_dev()
    lrs = args.lrs or cfg["learning_rates"]
    seeds = args.seeds or cfg["seeds"]
    grid = {}
    for lr in lrs:
        runs = [run_and_save(dev, cfg, lr, s, cfg["emoji_mode"], device, "cv", **fns) for s in seeds]
        grid[f"{lr:g}"] = summarize(runs)
    chosen = select(grid, cfg["tie_tolerance"])
    summary = {"selection_rule": "highest mean CV macro-F1 (mean over seeds of per-fold mean); "
                                 f"ties within {cfg['tie_tolerance']} broken by lower mean fold SD",
               "emoji_mode": cfg["emoji_mode"], "grid": grid, "selected_learning_rate": float(chosen),
               "test_exposure": "none"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cv_grid.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({lr: round(s["mean_cv_macro_f1"], 4) for lr, s in grid.items()}),
          "-> selected lr", chosen)
    return summary


def selected_lr():
    path = OUT / "cv_grid.json"
    if not path.exists():
        sys.exit("Run `python src/proposed_finetune.py cv` first; it selects the learning rate.")
    return json.loads(path.read_text())["selected_learning_rate"]


def cmd_ablation(cfg, args, device, **fns):
    dev, lr = load_dev(), selected_lr()
    seeds = args.seeds or cfg["seeds"]
    summary = {"learning_rate": lr, "full_method_emoji_mode": cfg["emoji_mode"], "conditions": {}}
    full = [json.loads((OUT / "cv" / f"P_{cfg['emoji_mode']}_lr{lr:g}_seed{s}_metrics.json").read_text())
            for s in seeds]
    summary["conditions"][cfg["emoji_mode"]] = summarize(full)
    for mode in args.modes or cfg["ablation_emoji_modes"]:
        runs = [run_and_save(dev, cfg, lr, s, mode, device, "ablation", **fns) for s in seeds]
        summary["conditions"][mode] = summarize(runs)
    (OUT / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({m: round(s["mean_cv_macro_f1"], 4) for m, s in summary["conditions"].items()}))
    return summary


def score_test(test, probs, meta, name):
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    preds = prediction_frame(test, probs, fold=-1)
    preds.to_csv(FINAL_OUT / f"{name}_test_predictions.csv", index=False)
    result = {**meta, "partition": "test", "n": len(test),
              **score(preds.y_true, preds.y_pred),
              "binary_pos_neg_vs_B0": binary_pos_neg_accuracy(preds.y_true, probs)}
    (FINAL_OUT / f"{name}_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: test macro-F1 {result['macro_f1']:.4f} | accuracy {result['accuracy']:.4f}")
    return result


def cmd_final(cfg, args, device, train_fn=train_model, predict_fn=predict_probs):
    test = load_test()  # raises unless FINAL_EVAL=1
    dev, lr = load_dev(), selected_lr()
    seeds = args.seeds or cfg["seeds"]
    for mode in [cfg["emoji_mode"], *(args.modes or cfg["ablation_emoji_modes"])]:
        for seed in seeds:
            started = perf_counter()
            tok, model, log = train_fn(dev.text.map(EMOJI[mode]).tolist(), dev.sentiment.tolist(),
                                       cfg, lr, seed, device)
            train_time = perf_counter() - started
            probs = predict_fn(tok, model, test.text.map(EMOJI[mode]).tolist(), cfg, device)
            name = f"P_{mode}_seed{seed}"
            score_test(test, probs, {"method_id": f"P-{mode}", "run_id": name, "seed": seed,
                                     "learning_rate": lr, "emoji_mode": mode,
                                     "trained_on": "all 521 dev posts", "config": cfg,
                                     "train_log": log, "train_seconds": round(train_time, 1),
                                     "hardware": hardware(device), "versions": versions()}, name)
            if mode == cfg["emoji_mode"] and seed == seeds[0] and hasattr(model, "save_pretrained"):
                path = CKPT / f"proposed_{mode}_seed{seed}"
                model.save_pretrained(path)
                tok.save_pretrained(path)
                (path / "training_record.json").write_text(json.dumps(
                    {"learning_rate": lr, "seed": seed, "emoji_mode": mode, "config": cfg,
                     "train_log": log}, indent=2))
                print(f"Saved checkpoint to {path}")
            del model


def cmd_eval_checkpoint(cfg, args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    test = load_test()
    path = Path(args.path)
    record = json.loads((path / "training_record.json").read_text())
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    probs = predict_probs(tok, model, test.text.map(EMOJI[record["emoji_mode"]]).tolist(), cfg, device)
    return score_test(test, probs, {"method_id": f"P-{record['emoji_mode']}", "seed": record["seed"],
                                    "evaluated": "saved checkpoint, no retraining",
                                    "checkpoint": str(path), "hardware": hardware(device)},
                      f"checkpoint_{path.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["cv", "ablation", "final", "eval-checkpoint"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--lrs", type=float, nargs="+")
    ap.add_argument("--modes", nargs="+", choices=["removed", "described"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--path", help="checkpoint folder for eval-checkpoint")
    args = ap.parse_args(argv)
    cfg = json.loads(CONFIG.read_text())
    device = pick_device(args.device)
    print(f"device={device}", flush=True)
    return {"cv": cmd_cv, "ablation": cmd_ablation, "final": cmd_final,
            "eval-checkpoint": cmd_eval_checkpoint}[args.command](cfg, args, device)


if __name__ == "__main__":
    main()
