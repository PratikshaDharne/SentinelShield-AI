"""Shared context for all experiments: models, record sets, policies, and a cached policy x dataset metric grid."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from api.pipeline import Guardrail
from evaluation.features import load_rows, analyze_rows, Rec
from evaluation.metrics import evaluate, breakdowns
from evaluation.shift_sets import all_shift_sets
from policy.baselines import (KeywordPolicy, ClassifierOnly, RetrievalOnly, RetrievalAssisted, Hybrid)
from policy.rules.rule_policy import RulePolicy
from policy.supervised.model import SupervisedPolicy
from policy.q_learning.agent import QPolicy
from configs.settings import settings

MAIN_SET = "final_test"          # official held-out set (fresh seed, de-duplicated)
SETS = ["final_test", "unseen_test", "shift_obfuscation", "shift_paraphrase", "shift_context"]
if (settings.datasets_dir / "data" / "real_test.jsonl").exists():
    SETS.append("real_test")          # held-out REAL public-dataset prompts (family-disjoint from training)


class LlamaGuardPolicy:
    name = "llama_guard3"
    def __init__(self, preds): self.preds = preds
    def act(self, rec): return self.preds.get(rec.row["id"], "ALLOW")


class AllowBlockHybrid:
    """Hybrid policy with the REWRITE action disabled (Allow/Block only) for experiment 2."""
    name = "hybrid_allow_block_only"
    def __init__(self, inner): self.inner = inner
    def act(self, rec):
        a = self.inner.act(rec); return "BLOCK" if a == "REWRITE" else a


@dataclass
class Ctx:
    g: Guardrail
    recs: dict
    policies: dict
    grid: dict = field(default_factory=dict)
    q: object = None

    def run(self, pname: str, sname: str) -> dict:
        k = (pname, sname)
        if k not in self.grid:
            pol, recs = self.policies[pname], self.recs[sname]
            acts = [pol.act(r) for r in recs]
            m = evaluate(self.g, recs, acts, pname); m["set"] = sname
            m["breakdowns"] = breakdowns(self.g, recs, acts); m["_actions"] = acts
            self.grid[k] = m
        return self.grid[k]

    def public(self, m: dict) -> dict:
        return {k: v for k, v in m.items() if not k.startswith("_")}


def build_ctx() -> Ctx:
    g = Guardrail.get(); g.reload_policy()
    recs = {MAIN_SET: analyze_rows(load_rows("final_test"), g), "unseen_test": analyze_rows(load_rows("unseen_test"), g)}
    if "real_test" in SETS:
        recs["real_test"] = analyze_rows(load_rows("real_test"), g)
    for name, rows in all_shift_sets().items():
        recs[name] = analyze_rows(rows, g)
    q = QPolicy.load()
    pol = {"keyword_rules": KeywordPolicy(), "classifier_only": ClassifierOnly(), "retrieval_only": RetrievalOnly(),
           "retrieval_assisted": RetrievalAssisted(), "rules_3way": RulePolicy(), "hybrid_base": Hybrid(None, "hybrid_base")}
    sup = SupervisedPolicy.load()
    if sup:
        pol["supervised_arb"] = sup; pol["hybrid_sup"] = Hybrid(None, "hybrid_sup", sup.model and sup)
    if q: pol["q_learning"] = Hybrid(q, "q_learning", sup)
    lg = settings.results_dir / "llamaguard_predictions.json"
    if lg.exists(): pol["llama_guard3"] = LlamaGuardPolicy(json.loads(lg.read_text()))
    pol["hybrid_allow_block_only"] = AllowBlockHybrid(pol["hybrid_base"])
    try:
        from policy.q_learning.dqn import DQNAgent
        from policy.q_learning.live import DQNEvalPolicy, RL_DIR
        ag = DQNAgent.load(RL_DIR / "dqn_init.npz")
        if ag is not None and sup is not None: pol["dqn_rl"] = DQNEvalPolicy(ag, sup)
    except Exception:
        pass
    return Ctx(g, recs, pol, q=q)
