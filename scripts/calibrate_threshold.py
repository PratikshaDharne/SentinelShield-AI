"""Calibrate INTENT_SIM_THRESHOLD for the ACTIVE encoder (MiniLM similarities differ from the hash fallback).
Measures similarity between true intent and our rewrites on validation attack rows and recommends a threshold
(5th percentile, so ~95% of good rewrites pass) - rewrites below it are blocked by the validator.
  python scripts/calibrate_threshold.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.pipeline import Guardrail
from api.normalizer import normalize
from evaluation.features import load_rows

g = Guardrail.get(); sims = []
for r in load_rows("val") + load_rows("val_unseen"):
    if r["decision"] != "REWRITE" or not r.get("intent"):
        continue
    out = g.rewriter.rewrite(normalize(r["text"]))
    if out.status == "REWRITE":
        sims.append(g.enc.similarity(out.rewritten_prompt, r["intent"]))
sims = np.array(sims)
print(f"encoder={g.enc.kind} rewrites={len(sims)} mean={sims.mean():.3f} p5={np.percentile(sims, 5):.3f} p50={np.median(sims):.3f}")
rec = round(float(np.percentile(sims, 5)) - 0.02, 2)
print(f"\nRecommended: INTENT_SIM_THRESHOLD={max(rec, 0.3)}   (put it in .env)")
