"""One-command reproduction of the October pipeline.

    python run_all.py                  # clean -> verify committed split -> baselines
    python run_all.py --remake-split   # also regenerate splits/split.csv first
    python run_all.py --skip-b2        # skip the TweetEval model (no download)
"""
import argparse, os, subprocess, sys

PY = sys.executable


def run(script, out_file=None):
    print(f"\n=== {script} ===", flush=True)
    res = subprocess.run([PY, script], capture_output=bool(out_file), text=True)
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
    ap.add_argument("--remake-split", action="store_true")
    ap.add_argument("--skip-b2", action="store_true")
    args = ap.parse_args()

    run("src/data_cleaning.py")
    if args.remake_split:
        run("src/make_split.py")
    run("src/check_split.py", out_file="evidence/split_check_output.txt")
    if os.path.exists("src/baseline_logreg.py"):
        run("src/baseline_logreg.py")
    else:
        print("\n(src/baseline_logreg.py not found yet; skipping B1)")
    if not args.skip_b2:
        run("src/baseline_tweeteval.py")
    run("src/collect_results.py")
    run("src/error_slice.py")
    print("\nAll steps finished. Evidence files are in evidence/.")
