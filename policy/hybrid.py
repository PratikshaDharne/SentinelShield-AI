"""Hybrid Allow/Rewrite/Block policy.
Order of authority (strongest first):
  1. HARD RULES  (deterministic; the learned policy can never override a hard block)
  2. Rule/clause evidence + trained clause classifier
  3. Whole-prompt classifier + retrieval risk
  4. Optional learned Q-policy (Stage 2) - may only choose among SAFE actions (see allowed_actions)
"""
from __future__ import annotations
from dataclasses import dataclass, field

ACTIONS = ["ALLOW", "REWRITE", "BLOCK"]


@dataclass
class PolicyInput:
    rule_codes: list[str]
    hard_block: bool
    has_attack_clause: bool
    has_task: bool
    clf_attack_prob: float
    clf_category: str
    ret_attack_vote: float
    ret_top_sim: float
    context_risk: bool
    obfuscated: bool
    tool_sensitivity: str = "none"
    rewrite_confidence: float = 0.0
    rewritability: int = 1          # 0 no legit task found, 1 task found, 2 task found AND topic known to our KB
    cluster: int = 0                # semantic cluster id (MiniLM/hash embedding -> KMeans)
    prev_blocks: int = 0            # recent BLOCK decisions for this client (decision history)
    vec: list | None = None         # feature vector for the supervised policy
    sup_action: int = 0             # supervised policy's recommendation (0 ALLOW,1 REWRITE,2 BLOCK) - state feature for Q


@dataclass
class PolicyOutput:
    action: str
    reason_codes: list[str]
    risk: float
    confidence: float
    needs_review: bool = False
    source: str = "hybrid"


def risk_score(p: PolicyInput, rule_sev: float) -> float:
    model = 0.65 * p.clf_attack_prob + 0.35 * p.ret_attack_vote
    ctx = 0.25 if p.context_risk else 0.0
    obf = 0.05 if p.obfuscated else 0.0
    return round(min(1.0, max(rule_sev, model) + ctx * (1 - max(rule_sev, model)) + obf), 4)


def allowed_actions(p: PolicyInput) -> list[str]:
    """Safety envelope for any learned policy."""
    if p.hard_block:
        return ["BLOCK"]
    if p.has_attack_clause:
        return ["REWRITE", "BLOCK"] if p.has_task else ["BLOCK"]
    if p.clf_attack_prob >= 0.90:                       # very confident model flag: never plain ALLOW
        return ["REWRITE", "BLOCK"]
    return list(ACTIONS) if (p.clf_attack_prob >= 0.35 or p.context_risk) else ["ALLOW", "REWRITE"]


class HybridPolicy:
    def __init__(self, q_policy=None, block_threshold: float = 0.90, review_threshold: float = 0.40, sup=None):
        self.q, self.sup = q_policy, sup
        self.block_t, self.review_t = block_threshold, review_threshold

    def decide(self, p: PolicyInput, rule_sev: float = 0.0) -> PolicyOutput:
        risk = risk_score(p, rule_sev)
        if p.hard_block:
            return PolicyOutput("BLOCK", ["HARD_BLOCK"] + p.rule_codes, 1.0, 1.0, False, "hard_rule")
        if p.has_attack_clause:
            if p.has_task:
                base = PolicyOutput("REWRITE", ["ATTACK_SPAN_WITH_LEGIT_TASK"] + p.rule_codes, risk, 0.9)
            else:
                base = PolicyOutput("BLOCK", ["NO_LEGIT_INTENT"] + p.rule_codes, risk, 0.95)
        elif p.context_risk and (p.clf_attack_prob >= 0.35 or p.ret_attack_vote >= 0.5):
            base = PolicyOutput("BLOCK", ["CONTEXT_MANIPULATION"], risk, 0.8, True)
        elif risk >= self.block_t:
            base = PolicyOutput("BLOCK", ["MODEL_HIGH_RISK", p.clf_category.upper()], risk, risk, True, "model")
        elif risk >= self.review_t:
            base = PolicyOutput("ALLOW", ["AMBIGUOUS_ALLOWED_FOR_REVIEW"], risk, 1 - risk, True, "model")
        else:
            base = PolicyOutput("ALLOW", ["BENIGN"], risk, 1 - risk)
        if base.source != "hard_rule":
            legal = allowed_actions(p); choice, via = None, None
            if self.sup is not None and p.vec is not None:
                try: p.sup_action = ACTIONS.index(self.sup.act_vec(p.vec))
                except Exception: pass
            if self.q is not None:
                try: choice, via = self.q.choose(p, risk), "q_learning"
                except Exception: choice = None
            if choice is None and self.sup is not None and p.vec is not None and not p.has_attack_clause:
                try: choice, via = self.sup.act_vec(p.vec), "supervised"
                except Exception: choice = None
            if choice in legal and choice != base.action:
                return PolicyOutput(choice, [f"LEARNED_{via.upper()}_{choice}"] + base.reason_codes, risk, base.confidence, True, via)
        return base
