"""Build the versioned benchmark: synthetic generator + (optional) real datasets downloaded by
scripts/download_datasets.py. Writes datasets/data/{train,val,test,unseen_test}.jsonl and a manifest."""
import json, re, sys, hashlib, random
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.synth import generate
from policy.rules.hard_rules import scan

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "data"; OUT.mkdir(parents=True, exist_ok=True)
EXT = ROOT / "datasets" / "external"


def split_of(text: str) -> str:
    r = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
    return "train" if r < 70 else "val" if r < 85 else "test"


def load_external():
    """Real public datasets (datasets/external/*.jsonl). Near-duplicate templates share a `family` so they never straddle splits.
    Benign-labelled real prompts that contain an explicit override/jailbreak clause are 'disputed' (the source labelled them
    regular, our policy would REWRITE them): they are written to disputed.jsonl and excluded from training AND metrics."""
    rows, disputed = [], []
    for f in sorted(EXT.glob("*.jsonl")) if EXT.exists() else []:
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            t = (d.get("text") or "").strip()
            if len(t) < 4:
                continue
            cat = d["category"]
            dec = d.get("decision") or ("ALLOW" if cat == "benign" else "BLOCK" if cat in ("harmful", "tool_abuse") else "REWRITE")
            fam = re.sub(r"[^a-z0-9]", "", t.lower())[:30]
            row = {"id": "x" + hashlib.md5(t.encode()).hexdigest()[:10], "text": t[:1500], "category": cat, "decision": dec,
                   "reason_codes": [d.get("source", "EXTERNAL").upper()], "split": split_of(fam), "source": d.get("source", f.stem),
                   "unseen": False, "history": [], "has_task": None, "task_topic": None, "family": "ext:" + fam}
            if dec == "ALLOW" and scan(row["text"]):
                disputed.append(row); continue
            rows.append(row)
    return rows, disputed


def main():
    rows = generate()
    ext, disputed = load_external()
    print(f"synthetic={len(rows)} external={len(ext)} disputed(excluded)={len(disputed)}")
    rows += ext
    for sp in ("train", "val", "val_unseen", "test", "unseen_test"):
        with (OUT / f"{sp}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                if r["split"] == sp:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "disputed.jsonl").open("w", encoding="utf-8") as fh:
        for r in disputed:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # REAL held-out set: external (public-dataset) rows whose family hashed into the test split
    real_test = [dict(r, split="real_test") for r in ext if r["split"] == "test"]
    with (OUT / "real_test.jsonl").open("w", encoding="utf-8") as fh:
        for r in real_test:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"real_test (public datasets, family-disjoint from train): {len(real_test)}")
    # fresh official held-out split: different RNG seed, de-duplicated against every other split
    seen = {r["text"] for r in rows}
    fresh = [r for r in generate(seed=9001) if r["split"] == "test" and r["text"] not in seen]
    with (OUT / "final_test.jsonl").open("w", encoding="utf-8") as fh:
        for r in fresh:
            r["split"] = "final_test"; fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"final_test (fresh seed, de-duplicated): {len(fresh)}")
    manifest = {"version": "1.0.0", "counts": {**Counter(r["split"] for r in rows), "final_test": len(fresh), "real_test": len(real_test), "disputed_excluded": len(disputed)}, "by_category": Counter(r["category"] for r in rows),
                "by_decision": Counter(r["decision"] for r in rows), "by_source": Counter(r["source"] for r in rows)}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
