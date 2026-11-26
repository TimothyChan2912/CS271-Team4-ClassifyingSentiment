"""One-command reproduction of the study.

    python run_all.py                  # clean -> split -> baselines/ablations -> summary (dev)
    python run_all.py --datasize 20000 # use up to 20,000 raw CSV records before cleaning
    python run_all.py --skip-b2        # skip the TweetEval model (no download)
    python run_all.py --proposed       # also: P learning-rate grid + emoji ablation (dev)
    python run_all.py --proposed --final
        # also: one-time test evaluation of every method, results table, figures
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

from src.training_progress import training_run_counts

PY = sys.executable


def run(script, *args, out_file=None, env=None):
    print(f"\n=== {script} {' '.join(args)} ===", flush=True)
    res = subprocess.run([PY, script, *args], capture_output=bool(out_file), text=True,
                         **({"env": env} if env else {}))
    if out_file:
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode == 0:
            with open(out_file, "w") as f:
                f.write(res.stdout)
    if res.returncode != 0:
        sys.exit(f"FAILED: {script} (exit {res.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasize", type=int, default=10000,
                    help="maximum raw YouTube CSV records to read before cleaning (default: 10000)")
    ap.add_argument("--remake-split", action="store_true",
                    help="accepted for compatibility; the split is now regenerated on every run")
    ap.add_argument("--skip-b2", action="store_true")
    ap.add_argument("--proposed", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.datasize <= 0:
        ap.error("--datasize must be a positive integer")

    run("src/data_cleaning.py", "--datasize", str(args.datasize))
    # A different input size changes IDs/partitions, so never reuse a stale split.
    run("src/make_split.py")
    run("src/check_split.py", out_file="evidence/split_check_output.txt")
    if os.path.exists("src/baseline_logreg.py"):
        run("src/baseline_logreg.py")
        run("src/run_b1_emoji_ablation.py")
    else:
        print("\n(src/baseline_logreg.py not found yet; skipping B1)")
    if not args.skip_b2:
        run("src/baseline_tweeteval.py")
        run("src/run_b2_emoji_ablation.py")
    proposed_envs = {}
    if args.proposed or args.final:
        cfg = json.loads((Path(__file__).resolve().parent / "configs/proposed_finetune.json").read_text())
        counts = training_run_counts(cfg)
        stages = (["cv", "ablation"] if args.proposed else []) + (["final"] if args.final else [])
        total, offset = sum(counts[stage] for stage in stages), 0
        for stage in stages:
            proposed_envs[stage] = {**os.environ, "PROPOSED_TRAINING_TOTAL": str(total),
                                    "PROPOSED_TRAINING_OFFSET": str(offset)}
            offset += counts[stage]
    if args.proposed:
        run("src/proposed_finetune.py", "cv", env=proposed_envs["cv"])
        run("src/proposed_finetune.py", "ablation", env=proposed_envs["ablation"])
    if args.final:
        final_env = {**os.environ, "FINAL_EVAL": "1"}
        run("src/proposed_finetune.py", "final", env={**proposed_envs["final"], "FINAL_EVAL": "1"})
        run("src/final_eval.py", *(["--skip-b2"] if args.skip_b2 else []), env=final_env)
    summary_args = []
    if args.proposed or args.final:
        summary_args.append("--proposed")
    if args.final:
        summary_args.append("--final")
    if args.skip_b2:
        summary_args.append("--skip-b2")
    run("src/run_summary.py", *summary_args, env=final_env if args.final else None)
    if args.final:
        run("src/make_figures.py", *(["--skip-b2"] if args.skip_b2 else []))
    print("\nAll steps finished. Evidence files are in evidence/.")
