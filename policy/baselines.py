"""Baseline policies required by the PDF (section 10). Each exposes name + act(rec) -> ALLOW|REWRITE|BLOCK.
 1 keyword        2 classifier-only      3 llama_guard (offline-only baseline, see evaluation/)
 4 retrieval      5 supervised (policy/supervised/model.py)   6 q_learning   7 hybrid"""
from __future__ import annotations
from policy.hybrid import HybridPolicy

KEYWORDS = ["ignore previous", "ignore all previous", "ignore your instructions", "ignore the above", "disregard", "system prompt",
            "reveal your", "jailbreak", "developer mode", "you are now", "pretend", "act as", "no restrictions", "bypass", "override",
            "disable", "hidden prompt", "forget your", "forget everything", "do anything now", "dan ", "unrestricted", "uncensored",
            "confidential instructions", "secret prompt", "obey only me", "admin override", "rm -rf", "drop table", "ransomware", "bomb"]


class KeywordPolicy:
    name = "keyword_rules"
    def act(self, rec) -> str:
        t = rec.row["text"].lower()
        return "BLOCK" if any(k in t for k in KEYWORDS) else "ALLOW"


class ClassifierOnly:
    name = "classifier_only"
    def __init__(self, thr: float = 0.5): self.thr = thr
    def act(self, rec) -> str:
        return "BLOCK" if rec.an.attack_p >= self.thr else "ALLOW"


class RetrievalOnly:
    name = "retrieval_only"
    def __init__(self, thr: float = 0.5): self.thr = thr
    def act(self, rec) -> str:
        return "BLOCK" if rec.an.ret.get("ret_attack_vote", 0.0) >= self.thr else "ALLOW"


class RetrievalAssisted:
    """MiniLM + FAISS retrieval combined with the classifier score (Allow/Block)."""
    name = "retrieval_assisted"
    def __init__(self, thr: float = 0.5): self.thr = thr
    def score(self, rec) -> float:
        return 0.5 * rec.an.attack_p + 0.5 * rec.an.ret.get("ret_attack_vote", 0.0)
    def act(self, rec) -> str:
        return "BLOCK" if self.score(rec) >= self.thr else "ALLOW"


class Hybrid:
    """Production policy: hard rules + clause evidence + classifier + retrieval (+ optional learned Q-policy)."""
    def __init__(self, q=None, name="hybrid", sup=None):
        self.p = HybridPolicy(q_policy=q, sup=sup); self.name = name
    def act(self, rec) -> str:
        return self.p.decide(rec.an.policy_input(), rec.an.rule_sev).action


class Masked:
    """Ablation wrapper: hide one signal group from the hybrid policy."""
    def __init__(self, drop: str, q=None, sup=None):
        self.drop, self.p, self.name = drop, HybridPolicy(q_policy=q, sup=sup), f"hybrid_minus_{drop}"
    def act(self, rec) -> str:
        pin = rec.an.policy_input(); sev = rec.an.rule_sev
        if self.drop == "rules":
            pin.rule_codes, pin.has_attack_clause, pin.hard_block, sev = [], False, False, 0.0
            if pin.vec: pin.vec = list(pin.vec); pin.vec[0] = pin.vec[1] = pin.vec[2] = 0.0
        elif self.drop == "classifier":
            pin.clf_attack_prob, pin.clf_category = 0.0, "benign"
            if pin.vec: pin.vec = list(pin.vec); pin.vec[4:12] = [0.0] * 8
        elif self.drop == "retrieval":
            pin.ret_attack_vote, pin.ret_top_sim = 0.0, 0.0
            if pin.vec: pin.vec = list(pin.vec); pin.vec[12:16] = [0.0] * 4
        elif self.drop == "context":
            pin.context_risk, pin.prev_blocks = False, 0
            if pin.vec: pin.vec = list(pin.vec); pin.vec[16] = 0.0
        elif self.drop == "semantic":     # embeddings: retrieval + semantic cluster
            pin.ret_attack_vote, pin.ret_top_sim, pin.cluster = 0.0, 0.0, 0
            if pin.vec: pin.vec = list(pin.vec); pin.vec[12:16] = [0.0] * 4
        return self.p.decide(pin, sev).action
