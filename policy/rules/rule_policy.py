"""Structural rule policy (no learned models): hard rules + clause tagger -> Allow / Rewrite / Block."""
from __future__ import annotations


class RulePolicy:
    name = "rules_3way"

    def act(self, rec) -> str:
        an = rec.an
        rule_hits = [h for h in an.tagged.attack_hits if h.code != "MODEL_FLAG"]
        if an.hard_block:
            return "BLOCK"
        if rule_hits or an.ctx_risk and an.rule_codes:
            return "REWRITE" if an.tagged.has_task and rule_hits else "BLOCK"
        return "ALLOW"
