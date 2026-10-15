"""E1 - Does adaptive decision-making improve over fixed rules?"""
from experiments.common import SETS

def run(c):
    pols = [p for p in ("keyword_rules", "rules_3way", "supervised_arb", "hybrid_base", "hybrid_sup", "q_learning") if p in c.policies]
    t = {p: {s: c.public(c.run(p, s)) for s in SETS} for p in pols}
    pick = lambda p, s, k: t[p][s][k]
    ans = [f"On {s}: rules_3way accuracy {pick('rules_3way', s, 'decision_accuracy')}, recall {pick('rules_3way', s, 'security_recall')} vs "
           f"learned (supervised) {pick('supervised_arb', s, 'decision_accuracy')}/{pick('supervised_arb', s, 'security_recall')} vs "
           f"hybrid {pick('hybrid_base', s, 'decision_accuracy')}/{pick('hybrid_base', s, 'security_recall')}" for s in SETS if 'supervised_arb' in t]
    return {"question": "Does adaptive decision-making improve over fixed rules?", "results": t, "summary": ans}
