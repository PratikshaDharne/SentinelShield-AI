"""Stage-1 held-out evaluation of the full guard (rules + classifier + retrieval + rewrite agent).
Reports decision accuracy, security recall, false-positive rate, safety-constraint violations,
rewrite success and latency percentiles on the TEST and UNSEEN splits. Never touches train/val."""
import json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.pipeline import Guardrail
from configs.settings import settings


def run(split, g):
    rows = [json.loads(l) for l in (settings.datasets_dir / "data" / f"{split}.jsonl").open(encoding="utf-8")]
    y, p, lat, rw_ok, rw_n, cat = [], [], [], 0, 0, Counter()
    for r in rows:
        res = g.evaluate(r["text"], r.get("history", [])[-6:])
        y.append(r["decision"]); p.append(res.decision); lat.append(res.latency_ms)
        if r["decision"] == "REWRITE":
            rw_n += 1; rw_ok += res.decision == "REWRITE"
        cat[(r["category"], r["decision"] == res.decision)] += 1
    y, p = np.array(y), np.array(p)
    risky, benign = y != "ALLOW", y == "ALLOW"
    out = {"split": split, "n": len(y), "decision_accuracy": round(float((y == p).mean()), 4),
           "security_recall": round(float((p[risky] != "ALLOW").mean()), 4),
           "safety_constraint_violation_rate": round(float((p[risky] == "ALLOW").mean()), 4),
           "false_positive_rate": round(float((p[benign] != "ALLOW").mean()), 4),
           "block_rate_on_benign": round(float((p[benign] == "BLOCK").mean()), 4),
           "rewrite_recall_on_rewrite_cases": round(float((p[y == "REWRITE"] == "REWRITE").mean()), 4) if (y == "REWRITE").any() else None,
           "latency_ms": {k: round(float(np.percentile(lat, q)), 2) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
           "confusion": {a: {b: int(((y == a) & (p == b)).sum()) for b in ("ALLOW", "REWRITE", "BLOCK")} for a in ("ALLOW", "REWRITE", "BLOCK")},
           "encoder": g.enc.kind}
    return out


if __name__ == "__main__":
    g = Guardrail.get(); res = [run(s, g) for s in ("test", "unseen_test")]
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "stage1_quick_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
