"""One command to (re)build every model on YOUR machine with whatever encoder is available.
  python scripts/setup_models.py            # real data -> classifier -> supervised -> Q-policy -> live DQN -> rewrite pairs -> evaluation
  python scripts/setup_models.py --skip-eval --skip-q     # faster
If sentence-transformers is installed, MiniLM is used automatically (backbone 'minilm', MiniLM+FAISS retrieval)."""
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(*args):
    print("\n>>>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(args)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-q", action="store_true"); ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="skip downloading the real public datasets (GitHub)")
    ap.add_argument("--with-rewriter", action="store_true", help="also fine-tune Flan-T5 (slow on CPU)")
    a = ap.parse_args()
    from embeddings.encoder import Encoder
    kind = Encoder.get().kind
    backbone = "minilm" if kind == "minilm" else "tfidf"
    print(f"encoder = {kind}  ->  classifier backbone = {backbone}")
    if not a.no_download:
        try: run("scripts/download_github_datasets.py")            # real jailbreak / harmful / user-prompt data (no HF login needed)
        except SystemExit: print("   (download skipped - no internet? continuing with whatever is in datasets/external)")
    run("scripts/build_dataset.py")
    run("scripts/train_classifier.py", "--backbone", backbone)
    run("scripts/train_supervised.py")
    if not a.skip_q:
        run("scripts/train_q_policy.py", "--epochs", "25")
    run("scripts/train_dqn.py")                                    # pre-train the live deep-RL agent (keeps learning online afterwards)
    run("scripts/calibrate_threshold.py")
    run("scripts/build_rewrite_dataset.py")
    if a.with_rewriter:
        run("scripts/train_rewriter.py", "--base", "google/flan-t5-small", "--epochs", "4")
    if not a.skip_eval:
        run("scripts/run_evaluation.py")
    print("\nDone. Start the stack: docker compose up -d --build   (or: bash scripts/run_api.sh + streamlit run dashboard/app.py)")


if __name__ == "__main__":
    main()
