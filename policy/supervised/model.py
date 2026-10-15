"""Supervised Allow/Rewrite/Block classifier over the same evidence the hybrid policy sees."""
from __future__ import annotations
import joblib, math
import numpy as np
from configs.settings import settings
from classifiers.threat_classifier import CATEGORIES

FEATURES = (["rule_sev", "hard_block", "has_attack_clause", "has_task", "attack_p"] + [f"p_{c}" for c in CATEGORIES] +
            ["ret_vote", "ret_top_sim", "ret_best_attack", "ret_best_benign", "ctx_risk", "obf", "log_len", "rewritability",
             "n_rule_codes", "n_model_flags"])


def vector(an) -> list[float]:
    ret = an.ret
    return ([an.rule_sev, float(an.hard_block), float(bool(an.tagged.attack_hits)), float(an.tagged.has_task), an.attack_p] +
            [an.probs.get(c, 0.0) for c in CATEGORIES] +
            [ret.get("ret_attack_vote", 0.0), ret.get("ret_top_sim", 0.0), ret.get("ret_best_attack_sim", 0.0),
             ret.get("ret_best_benign_sim", 0.0), float(an.ctx_risk), float(bool(an.norm.flags)), math.log1p(len(an.prompt)),
             float(an.rewritability), float(len(an.rule_codes)), float(sum(h.code == "MODEL_FLAG" for h in an.tagged.attack_hits))])


class SupervisedPolicy:
    name = "supervised_arb"

    def __init__(self, model=None): self.model = model

    def fit(self, recs):
        from sklearn.ensemble import GradientBoostingClassifier
        X = np.array([vector(r.an) for r in recs]); y = np.array([r.gt for r in recs])
        w = np.where(y == "ALLOW", 1.0, 1.5)
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0).fit(X, y, sample_weight=w)
        return self

    def act(self, rec) -> str:
        return str(self.model.predict(np.array([vector(rec.an)]))[0])

    def act_vec(self, vec) -> str:
        return str(self.model.predict(np.array([vec]))[0])

    def save(self, p=None): joblib.dump(self.model, p or settings.models_dir / "supervised_policy.joblib")

    @classmethod
    def load(cls, p=None):
        p = p or settings.models_dir / "supervised_policy.joblib"
        try: return cls(joblib.load(p))
        except Exception: return None
