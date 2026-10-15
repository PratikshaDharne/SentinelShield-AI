"""Download REAL public prompt datasets that are hosted on GitHub (no Hugging Face login needed) and convert them to
datasets/external/*.jsonl. Works on any machine with internet; also works behind proxies that only allow GitHub.

Sources (all public research datasets):
  wild_jailbreak   1,405 in-the-wild jailbreak prompts collected from Reddit / Discord / websites (verazuo/jailbreak_llms, Shen et al. 2023)
  wild_regular     13,735 real user prompts from the same communities (same repo) - benign class
  forbidden_q      390 forbidden questions across 13 policy areas (same repo)             -> harmful
  advbench         520 harmful behaviours (llm-attacks/llm-attacks, Zou et al. 2023)       -> harmful
  harmbench        400 harmful behaviours (centerforaisafety/HarmBench)                    -> harmful
  self_instruct    252 human-written, user-oriented instructions (yizhongw/self-instruct)  -> benign
  mt_bench         80 multi-turn benchmark questions (lm-sys/FastChat)                      -> benign
  alpaca_clean     cleaned Stanford Alpaca instructions (gururise/AlpacaDataCleaned)        -> benign (machine-generated)
  code_alpaca      code instructions (sahil280114/codealpaca)                               -> benign (machine-generated)

  python scripts/download_github_datasets.py            # defaults keep the training set manageable
  python scripts/download_github_datasets.py --regular 8000 --alpaca 6000
Then: python scripts/build_dataset.py && python scripts/train_classifier.py && python scripts/train_supervised.py ...
Honest note: these cover jailbreak, harmful and benign traffic. Prompt-injection / leak / tool-abuse real data come from
scripts/download_datasets.py (Hugging Face); until then those classes remain synthetic."""
import argparse, io, json, random, re, sys
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "external"; OUT.mkdir(parents=True, exist_ok=True)
RAW = "https://raw.githubusercontent.com/"
MAXLEN = 1500


def get(path: str) -> bytes:
    r = requests.get(RAW + path, timeout=120); r.raise_for_status(); return r.content


def clean(t) -> str:
    return " ".join(str(t or "").replace("\u0000", " ").split())


def write(name, rows):
    seen, n = set(), 0
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            t = clean(r["text"])
            if len(t) < 8 or t.lower() in seen:
                continue
            seen.add(t.lower()); r = {**r, "text": t[:MAXLEN], "source": name}
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"[ok] {name}: {n} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular", type=int, default=6000); ap.add_argument("--alpaca", type=int, default=4000)
    ap.add_argument("--code", type=int, default=1000)
    a = ap.parse_args(); rnd = random.Random(11)
    J = "verazuo/jailbreak_llms/main/data/"
    jb = pd.read_csv(io.BytesIO(get(J + "prompts/jailbreak_prompts_2023_12_25.csv")))
    # whole-prompt persona/jailbreak templates carry no legitimate task -> BLOCK is the correct decision
    write("wild_jailbreak", [{"text": p, "category": "jailbreak", "decision": "BLOCK"} for p in jb.prompt.astype(str)])
    rg = pd.read_csv(io.BytesIO(get(J + "prompts/regular_prompts_2023_12_25.csv"))).prompt.astype(str).tolist()
    rg = [p for p in rg if 8 <= len(clean(p)) <= MAXLEN]; rnd.shuffle(rg)
    write("wild_regular", [{"text": p, "category": "benign"} for p in rg[:a.regular]])
    fq = pd.read_csv(io.BytesIO(get(J + "forbidden_question/forbidden_question_set.csv")))
    write("forbidden_q", [{"text": q, "category": "harmful"} for q in fq.question])
    ab = pd.read_csv(io.BytesIO(get("llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv")))
    write("advbench", [{"text": q, "category": "harmful"} for q in ab.goal])
    hb = pd.read_csv(io.BytesIO(get("centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv")))
    write("harmbench", [{"text": q, "category": "harmful"} for q in hb.Behavior])
    si = [json.loads(l) for l in get("yizhongw/self-instruct/main/human_eval/user_oriented_instructions.jsonl").decode().splitlines() if l.strip()]
    write("self_instruct", [{"text": (d["instruction"] + " " + (d["instances"][0].get("input") or "")), "category": "benign"} for d in si])
    mt = [json.loads(l) for l in get("lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl").decode().splitlines() if l.strip()]
    write("mt_bench", [{"text": t, "category": "benign"} for d in mt for t in d["turns"]])
    al = json.loads(get("gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json")); rnd.shuffle(al)
    write("alpaca_clean", [{"text": (d["instruction"] + " " + d.get("input", "")), "category": "benign"} for d in al[:a.alpaca * 2]][:a.alpaca])
    ca = json.loads(get("sahil280114/codealpaca/master/data/code_alpaca_20k.json")); rnd.shuffle(ca)
    write("code_alpaca", [{"text": (d["instruction"] + " " + d.get("input", "")), "category": "benign"} for d in ca[:a.code]])


if __name__ == "__main__":
    sys.exit(main())
