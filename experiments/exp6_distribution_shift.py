"""E6 - Does the policy remain robust to unseen patterns? (policies x shift sets, plus per obfuscation family)"""
from collections import defaultdict
from experiments.common import SETS

def run(c):
    pols = [p for p in ("keyword_rules", "classifier_only", "retrieval_assisted", "rules_3way", "supervised_arb", "hybrid_base", "hybrid_sup", "q_learning", "llama_guard3") if p in c.policies]
    grid = {p: {s: {k: c.run(p, s)[k] for k in ("n", "decision_accuracy", "security_recall", "safety_constraint_violation_rate", "false_positive_rate")} for s in SETS} for p in pols}
    fam = {}
    for p in ("hybrid_sup", "hybrid_base", "q_learning", "classifier_only", "keyword_rules"):
        if p not in c.policies: continue
        m = c.run(p, "shift_obfuscation"); d = defaultdict(list)
        for rec, a in zip(c.recs["shift_obfuscation"], m["_actions"]):
            d[rec.row["source"].replace("shift_obf_", "")].append(a != "ALLOW")
        fam[p] = {k: round(sum(v) / len(v), 4) for k, v in d.items()}
    return {"question": "Does the policy remain robust to unseen patterns?", "grid": grid, "obfuscation_family_recall": fam}
