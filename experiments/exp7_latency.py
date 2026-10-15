"""E7 - Can the guardrail operate in real time? Full pipeline (incl. rewrite) single-thread + concurrent throughput."""
import time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from evaluation.features import load_rows

def run(c, n=600):
    rows = load_rows("final_test"); rows = (rows * (n // max(len(rows), 1) + 1))[:n]
    prompts = [r["text"] for r in rows]
    for p in prompts[:20]: c.g.evaluate(p)                      # warm-up
    lat, stages, by_dec = [], {}, {"ALLOW": [], "REWRITE": [], "BLOCK": []}
    for p in prompts:
        t = time.perf_counter(); res = c.g.evaluate(p, session_key="lat-bench"); ms = (time.perf_counter() - t) * 1000
        lat.append(ms); by_dec[res.decision].append(ms)
        for k, v in res.stage_latency_ms.items(): stages.setdefault(k, []).append(v)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda p: c.g.evaluate(p), prompts[:400]))
    rps = 400 / (time.perf_counter() - t0)
    pc = lambda a: {k: round(float(np.percentile(a, q)), 2) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))} if a else None
    return {"question": "Can the guardrail operate in real time?", "encoder": c.g.enc.kind, "n": len(lat), "overall_ms": pc(lat),
            "by_decision_ms": {k: pc(v) for k, v in by_dec.items()}, "mean_stage_ms": {k: round(float(np.mean(v)), 2) for k, v in stages.items()},
            "histogram_ms": [round(x, 2) for x in lat[:600]], "throughput_rps_8_threads": round(rps, 1),
            "note": "Pipeline latency (excludes network and the answer-LLM call). With MiniLM expect a few extra ms per request on CPU."}
