"""OPTIONAL and OFF BY DEFAULT. One-time OFFLINE dataset augmentation: asks Gemini to write extra clean rewrite targets
(more varied phrasing) for the rewriter's training pairs. Gemini is never used by the rewriter at runtime.
  python scripts/augment_with_gemini.py --enable --n 500      # needs GEMINI_API_KEY; writes datasets/rewrite_pairs_gemini.jsonl
Then: append those rows to datasets/rewrite_pairs_train.jsonl and re-run scripts/train_rewriter.py.
Every generated target is filtered through OUR validator + hard rules before it is kept."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.settings import settings
from validation.validator import Validator
from policy.rules.hard_rules import scan

PROMPT = ("Rewrite the following request as ONE clear, complete, natural prompt for an AI assistant. Keep the user's real goal, add helpful "
          "scope (key sub-topics), and write it as a single sentence or two. Output only the rewritten prompt.\nREQUEST: {task}\nSCOPE HINT: {scope}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--enable", action="store_true"); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    if not a.enable:
        print("Disabled by default. Re-run with --enable to call Gemini ONCE, offline, to create extra training targets."); return
    from api import llm
    v = Validator(); src = settings.datasets_dir / "rewrite_pairs_train.jsonl"; out = settings.datasets_dir / "rewrite_pairs_gemini.jsonl"
    kept = 0
    with out.open("w", encoding="utf-8") as fh:
        for line in src.open(encoding="utf-8"):
            if kept >= a.n: break
            d = json.loads(line)
            try:
                task = d["input"].split("task: ", 1)[1].split(" | ", 1)[0]; scope = d["input"].split("scope: ", 1)[1].split(" | ", 1)[0]
                text = llm._gemini([{"role": "user", "content": PROMPT.format(task=task, scope=scope)}], "Reply with the rewritten prompt only.").strip()
            except Exception as e:
                print("skip:", str(e)[:80]); time.sleep(2); continue
            if scan(text, False, strip_negations=True) or not v.validate(text, task).passed:
                continue
            fh.write(json.dumps({"input": d["input"], "target": text}, ensure_ascii=False) + "\n"); kept += 1; time.sleep(1.0)
    print(f"kept {kept} validated targets -> {out}")


if __name__ == "__main__":
    main()
