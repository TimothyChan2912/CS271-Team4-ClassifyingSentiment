"""B2: TweetEval RoBERTa sentiment model, zero-shot, on the dev partition.

PROTOCOL.md Section 7, B2:
    cardiffnlp/twitter-roberta-base-sentiment, zero-shot, max_length=128, truncation on

The model is not trained on our data, so its prediction for every dev post is
already a held-out prediction. Predictions are grouped by the existing fold
column so the per-fold mean/std is reported the same way as for B1.

Usage (from the repo root):
    python src/baseline_tweeteval.py
    python src/baseline_tweeteval.py --preprocess   # model-card preprocessing; NOT in protocol v1
"""
import argparse, json, os, platform, subprocess, sys, time

import numpy as np

from data_io import load_dev
from metrics import LABELS, binary_pos_neg_accuracy, cv_report

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MAX_LENGTH = 128
# Model card: LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive (same order as LABELS)
ID2SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}


def model_card_preprocess(text):
    """Preprocessing recommended on the model card (mentions -> @user, links -> http)."""
    out = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


def pick_device(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_commit(revision):
    """Exact Hugging Face commit of the model, so reruns can pin it with --revision."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(MODEL, revision=revision).sha
    except Exception:
        return None


def predict(texts, batch_size, device_arg, revision):
    """Return (probs (n,3) in LABELS order, run info)."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(42)  # inference is deterministic; set anyway
    device = pick_device(torch, device_arg)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=revision)
    assert model.config.num_labels == 3, "unexpected head size"
    model.to(device).eval()

    lengths = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    info = {
        "model": MODEL,
        "revision_requested": revision,
        "model_commit": getattr(model.config, "_commit_hash", None) or resolve_commit(revision),
        "device": device,
        "posts_truncated_at_128_tokens": int(sum(n > MAX_LENGTH for n in lengths)),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    return np.concatenate(probs), info


def hardware():
    """Short hardware description for the results row."""
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        except Exception:
            pass
    return f"{platform.system()} {platform.release()}, {cpu}"


def main(argv=None, predict_fn=predict):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    ap.add_argument("--revision", default="main",
                    help="Hugging Face commit to pin; the resolved commit is recorded either way")
    ap.add_argument("--preprocess", action="store_true",
                    help="apply model-card preprocessing (not part of protocol v1)")
    args = ap.parse_args(argv)

    dev = load_dev()
    texts = dev["text"].tolist()
    if args.preprocess:
        texts = [model_card_preprocess(t) for t in texts]

    t0 = time.perf_counter()
    probs, info = predict_fn(texts, args.batch_size, args.device, args.revision)
    runtime = time.perf_counter() - t0
    assert probs.shape == (len(dev), 3)

    dev["y_true"] = dev["sentiment"]
    dev["y_pred"] = [ID2SENTIMENT[i] for i in probs.argmax(axis=1)]
    for j, lab in enumerate(LABELS):
        dev[f"p_{lab}"] = probs[:, j]

    report = cv_report(dev[["fold", "y_true", "y_pred"]])
    report["binary_pos_neg_vs_B0"] = binary_pos_neg_accuracy(dev["y_true"], probs)
    report["config"] = {"id": "B2", "max_length": MAX_LENGTH, "truncation": True,
                        "zero_shot": True, "preprocess": args.preprocess,
                        "partition": "dev", "test_exposure": "none", "seed": "n/a (zero-shot, deterministic)",
                        "runtime_seconds": round(runtime, 1), "hardware": hardware(), **info}

    suffix = "_preprocessed" if args.preprocess else ""
    os.makedirs("evidence", exist_ok=True)
    cols = ["id", "fold", "y_true", "y_pred"] + [f"p_{lab}" for lab in LABELS]
    dev[cols].to_csv(f"evidence/b2_tweeteval_predictions{suffix}.csv", index=False)
    with open(f"evidence/b2_tweeteval_metrics{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)

    p = report["pooled"]
    print(f"B2 pooled macro-F1 {p['macro_f1']:.4f} | accuracy {p['accuracy']:.4f} | "
          f"per-fold macro-F1 {report['per_fold_macro_f1_mean']:.4f} "
          f"± {report['per_fold_macro_f1_std']:.4f}")
    print("per-class F1:", {k: round(v, 4) for k, v in p["per_class_f1"].items()})
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
