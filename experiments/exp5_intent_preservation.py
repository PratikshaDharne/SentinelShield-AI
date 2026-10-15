"""E5 - Can Rewrite maintain legitimate objectives? Measured against the TRUE intent (not the agent's own extraction)."""
import json, numpy as np
from experiments.common import SETS
from evaluation.features import rewrite_outcome, leaks_attack
from evaluation.metrics import INTENT_OK
from configs.settings import settings

def run(c):
    thr = INTENT_OK.get(c.g.enc.kind, 0.5); out = {}; pairs = []; before_all, after_all = [], []
    for s in SETS:
        rows = [r for r in c.recs[s] if r.gt == "REWRITE" and r.row.get("intent")]
        if not rows: continue
        rw = [(r, rewrite_outcome(c.g, r)) for r in rows]
        ok = [(r, o) for r, o in rw if o["ok"]]
        before = [o["sim_before"] for _, o in ok if o["sim_before"] is not None]; after = [o["sim_after"] for _, o in ok if o["sim_after"] is not None]
        before_all += before; after_all += after
        out[s] = {"n_rewrite_cases": len(rw), "rewrite_released": len(ok), "released_rate": round(len(ok) / len(rw), 4),
                  "unsafe_rate_among_released": round(float(np.mean([o["unsafe"] for _, o in ok])), 4) if ok else None,
                  "mean_similarity_before": round(float(np.mean(before)), 4) if before else None,
                  "mean_similarity_after": round(float(np.mean(after)), 4) if after else None,
                  "intent_preserved_rate": round(float(np.mean([(not o["unsafe"]) and (o["sim_after"] or 0) >= thr for _, o in ok])), 4) if ok else None,
                  "rewrite_success_rate_overall": round(sum((not o["unsafe"]) and (o["sim_after"] or 0) >= thr for _, o in ok) / len(rw), 4)}
        pairs += [{"id": r.row["id"], "intent": r.row["intent"], "rewrite": o["text"]} for r, o in ok[:60]]
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "judge_pairs.json").write_text(json.dumps(pairs[:300], indent=1))
    judge = settings.results_dir / "judge_scores.json"
    return {"question": "Can Rewrite maintain legitimate objectives?", "similarity_threshold": thr, "encoder": c.g.enc.kind, "per_set": out,
            "histogram": {"before": [round(x, 3) for x in before_all], "after": [round(x, 3) for x in after_all]},
            "llm_judge": json.loads(judge.read_text()) if judge.exists() else "not run (python scripts/run_evaluation.py --with-ollama)"}
