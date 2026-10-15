"""E4 - Does RL improve the decision trade-off? Includes a paired analysis of which decisions Q changed."""
from experiments.common import SETS
from evaluation.metrics import effective

def run(c):
    if "q_learning" not in c.policies:
        return {"question": "Does RL improve the decision trade-off?", "error": "q_learning policy not trained (run scripts/train_q_policy.py)"}
    ref = "hybrid_sup" if "hybrid_sup" in c.policies else "hybrid_base"      # Q is initialised from / compared against this
    res, paired = {}, {}
    for s in SETS:
        b, q = c.run(ref, s), c.run("q_learning", s)
        res[s] = {ref: c.public(b), "q_learning": c.public(q)}
        fixed = regressed = changed = 0; examples = []
        for rec, a0, a1 in zip(c.recs[s], b["_actions"], q["_actions"]):
            if a0 == a1: continue
            changed += 1
            e0, e1 = effective(c.g, rec, a0)[0], effective(c.g, rec, a1)[0]
            ok0, ok1 = e0 == rec.gt, e1 == rec.gt
            fixed += (not ok0) and ok1; regressed += ok0 and (not ok1)
            if len(examples) < 6: examples.append({"gt": rec.gt, "base": a0, "q": a1, "text": rec.row["text"][:110]})
        paired[s] = {"changed": changed, "fixed_by_q": int(fixed), "regressed_by_q": int(regressed), "examples": examples}
    return {"question": "Does RL improve the decision trade-off?", "reference_policy": ref, "q_meta": c.q.meta, "q_recommended": c.q.meta.get("recommended"), "results": res, "paired_changes": paired}
