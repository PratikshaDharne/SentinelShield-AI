"""Download large real-world datasets from Hugging Face and convert them to a common schema in
datasets/external/*.jsonl. RUN THIS ON YOUR MACHINE (needs internet). Some datasets are gated:
accept their terms on huggingface.co and run `huggingface-cli login` (or set HF_TOKEN) first.

  python scripts/download_datasets.py --benign-per-source 20000
Then: python scripts/build_dataset.py && python scripts/train_classifier.py
Each loader is wrapped in try/except, so a gated/unavailable dataset is skipped with a message."""
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "external"; OUT.mkdir(parents=True, exist_ok=True)


def first_user_turn(conv):
    for m in conv or []:
        role = m.get("role") or m.get("from")
        if role in ("user", "human"):
            return m.get("content") or m.get("value") or ""
    return ""


def write(name, rows):
    p = OUT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for t, cat in rows:
            t = " ".join((t or "").split())
            if 8 <= len(t) <= 1500:
                f.write(json.dumps({"text": t, "category": cat, "source": name}, ensure_ascii=False) + "\n")
    print(f"[ok] {name}: {p}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benign-per-source", type=int, default=20000)
    a = ap.parse_args()
    from datasets import load_dataset
    rnd = random.Random(7); N = a.benign_per_source

    def sample(ds, fn, n):
        idx = list(range(len(ds))); rnd.shuffle(idx)
        out = []
        for i in idx:
            r = fn(ds[i])
            if r: out.append(r)
            if len(out) >= n: break
        return out

    jobs = {
      # ---- real user traffic (benign) ----
      "wildchat": lambda: sample(load_dataset("allenai/WildChat-1M", split="train"),
                                 lambda r: (first_user_turn(r["conversation"]), "benign") if r.get("language") == "English" and not r.get("toxic") else None, N),
      "lmsys_chat": lambda: sample(load_dataset("lmsys/lmsys-chat-1m", split="train"),
                                   lambda r: (first_user_turn(r["conversation"]), "benign") if r.get("language") == "English" and not any(r.get("openai_moderation", [{}])[0].get("flagged", False) for _ in [0]) else None, N),
      "oasst1": lambda: sample(load_dataset("OpenAssistant/oasst1", split="train"),
                               lambda r: (r["text"], "benign") if r.get("role") == "prompter" and r.get("lang") == "en" else None, N),
      "dolly": lambda: sample(load_dataset("databricks/databricks-dolly-15k", split="train"), lambda r: (r["instruction"], "benign"), N),
      "alpaca": lambda: sample(load_dataset("tatsu-lab/alpaca", split="train"), lambda r: (r["instruction"], "benign"), N),
      # ---- injection / jailbreak / leak ----
      "deepset_pi": lambda: [(r["text"], "prompt_injection" if r["label"] == 1 else "benign")
                             for s in ("train", "test") for r in load_dataset("deepset/prompt-injections", split=s)],
      "safeguard_pi": lambda: [(r["text"], "prompt_injection" if r["label"] == 1 else "benign")
                               for s in ("train", "test") for r in load_dataset("xTRam1/safe-guard-prompt-injection", split=s)],
      "jackhhao_jb": lambda: [(r["prompt"], "jailbreak" if r["type"] == "jailbreak" else "benign")
                              for s in ("train", "test") for r in load_dataset("jackhhao/jailbreak-classification", split=s)],
      "inthewild_jb": lambda: [(r["prompt"], "jailbreak" if r.get("jailbreak") else "benign")
                               for r in load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train")],
      "toxicchat": lambda: [(r["user_input"], "jailbreak" if r.get("jailbreaking") == 1 else ("harmful" if r.get("toxicity") == 1 else "benign"))
                            for s in ("train", "test") for r in load_dataset("lmsys/toxic-chat", "toxicchat0124", split=s)],
      "gandalf_ignore": lambda: [(r["text"], "prompt_injection") for r in load_dataset("Lakera/gandalf_ignore_instructions", split="train")],
    }
    for name, fn in jobs.items():
        try:
            write(name, fn())
        except Exception as e:  # gated / renamed / offline
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
