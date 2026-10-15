"""E2 - Does rewriting reduce unnecessary blocking?"""
from experiments.common import SETS

def run(c):
    t = {p: {s: c.public(c.run(p, s)) for s in SETS} for p in ("hybrid_allow_block_only", "hybrid_base")}
    keys = ("legit_blocked_rate", "legit_utility", "security_recall", "false_positive_rate", "safety_constraint_violation_rate")
    ans = {s: {k: {"allow_block": t["hybrid_allow_block_only"][s][k], "allow_rewrite_block": t["hybrid_base"][s][k]} for k in keys} for s in SETS}
    return {"question": "Does rewriting reduce unnecessary blocking?", "results": t, "comparison": ans}
