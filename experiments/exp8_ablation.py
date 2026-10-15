"""E8 - Which signals contribute most to safe decisions? (hide one signal group at a time from the hybrid policy)"""
from experiments.common import SETS
from policy.baselines import Masked, Hybrid

DROPS = ["rules", "classifier", "retrieval", "context", "semantic"]

def run(c):
    from policy.supervised.model import SupervisedPolicy
    sup = SupervisedPolicy.load()
    for d in DROPS: c.policies[f"abl_{d}"] = Masked(d, None, sup)
    full = "hybrid_sup" if "hybrid_sup" in c.policies else "hybrid_base"
    names = {"full": full, **{d: f"abl_{d}" for d in DROPS}}
    out = {lab: {s: {k: c.run(p, s)[k] for k in ("decision_accuracy", "security_recall", "safety_constraint_violation_rate", "false_positive_rate")}
                 for s in SETS} for lab, p in names.items()}
    delta = {d: {s: round(out["full"][s]["decision_accuracy"] - out[d][s]["decision_accuracy"], 4) for s in SETS} for d in DROPS}
    return {"question": "Which signals contribute most to safe decisions?", "variants": out, "accuracy_drop_when_removed": delta,
            "groups": {"rules": "hard/structural rules + clause tagger", "classifier": "threat classifier", "retrieval": "MiniLM+FAISS vote",
                       "context": "multi-turn context + decision history", "semantic": "embedding-based retrieval + semantic cluster"}}
