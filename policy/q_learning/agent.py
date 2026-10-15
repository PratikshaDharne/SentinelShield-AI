"""Tabular Q-Learning policy.
 * Q-rows are initialised from the baseline action (hard rules + supervised policy) -> safe starting point (PDF 7)
 * trained ONLY offline in the simulator (policy/q_learning/env.py)
 * action masking: hard blocks / attack clauses restrict the legal actions (policy.hybrid.allowed_actions);
   the learned policy can never override a hard block
 * unseen / rarely visited states defer to the baseline (conservative)."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
from configs.settings import settings
from policy.hybrid import HybridPolicy, allowed_actions, ACTIONS
from policy.q_learning.state import make_state, N_STATES
from policy.q_learning.env import Lambdas, SessionSampler, reward

MIN_VISITS = 5
MIN_DEVIATION_VISITS = 15      # a state needs this much evidence on the better action before Q may override the baseline
QDIR = settings.models_dir / "q_policy"


class QPolicy:
    def __init__(self, Q: np.ndarray, visits: np.ndarray, meta: dict | None = None, margin: float | None = None):
        self.Q, self.visits, self.meta = Q, visits, meta or {}
        self.margin = float(margin if margin is not None else self.meta.get("margin", 0.0))

    def choose(self, p, risk: float):
        """Safe policy improvement: follow the baseline (the supervised recommendation) unless Q has clear, well-supported evidence
        that another LEGAL action is better by at least `margin`. Unseen/rare states defer entirely."""
        s = make_state(p, risk)
        if self.visits[s].sum() < MIN_VISITS:
            return None
        legal = [ACTIONS.index(a) for a in allowed_actions(p)]
        best = max(legal, key=lambda a: self.Q[s, a])
        ref = p.sup_action if p.sup_action in legal else None
        if ref is not None and best != ref and (self.Q[s, best] - self.Q[s, ref] < self.margin or self.visits[s, best] < MIN_DEVIATION_VISITS):
            best = ref
        return ACTIONS[best]

    def save(self, d: Path | None = None):
        d = Path(d or QDIR); d.mkdir(parents=True, exist_ok=True)
        self.meta["margin"] = self.margin
        np.savez_compressed(d / "q_table.npz", Q=self.Q, visits=self.visits)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, d: Path | None = None):
        d = Path(d or QDIR)
        f = d / "q_table.npz"
        if not f.exists():
            return None
        z = np.load(f)
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        return cls(z["Q"], z["visits"], meta)


def train(recs, rw_fn, lam: Lambdas, *, epochs: int = 30, gamma: float = 0.3, eps0: float = 0.35, eps1: float = 0.02,
          intent_floor: float = 0.5, seed: int = 0, log_every: int = 0, sup=None, sup_actions=None) -> tuple[QPolicy, list[float]]:
    """recs: training Rec list (OOF features); rw_fn(rec)->rewrite outcome dict (cached)."""
    rng = random.Random(seed)
    Q = np.zeros((N_STATES, 3), dtype=np.float32); N = np.zeros((N_STATES, 3), dtype=np.int32); init = np.zeros(N_STATES, bool)
    pins = [r.an.policy_input() for r in recs]
    if sup_actions is not None:
        for pin, sa in zip(pins, sup_actions): pin.sup_action = sa
    risks = [base_risk(r) for r in recs]
    # safe starting point: the (cross-fitted) supervised recommendation when legal, else the rules-only hybrid decision.
    # NOTE: never call a sup-aware policy here - its in-sample predictions would leak ground truth into training.
    rules_only = HybridPolicy(q_policy=None, sup=None)
    base_act = []
    for i, (p_, r_) in enumerate(zip(pins, recs)):
        legal_i = [ACTIONS.index(x) for x in allowed_actions(p_)]
        sa = sup_actions[i] if sup_actions is not None else None
        base_act.append(sa if sa is not None and sa in legal_i else ACTIONS.index(rules_only.decide(p_, r_.an.rule_sev).action))
    legal = [[ACTIONS.index(a) for a in allowed_actions(p)] for p in pins]
    S = [(make_state(p, rk, 0), make_state(p, rk, 1)) for p, rk in zip(pins, risks)]       # state for prev_blocks 0 / 1
    R = {}                                                                                  # reward cache per (rec, action)
    def rew(i, a):
        k = (i, a)
        if k not in R:
            R[k] = reward(ACTIONS[a], recs[i].gt, rw_fn(recs[i]) if a == 1 else None, lam, intent_floor)
        return R[k]
    sampler = SessionSampler(recs, seed)
    curve, n_sessions = [], max(len(recs) // 3, 500)
    for ep in range(epochs):
        eps = eps1 + (eps0 - eps1) * (1 - ep / max(epochs - 1, 1)); tot = 0.0; steps = 0
        for _ in range(n_sessions):
            sess = sampler.session(); pb = 0
            for t, i in enumerate(sess):
                s = S[i][pb]
                if not init[s]:
                    Q[s, base_act[i]] = 0.5; init[s] = True              # start at the safe baseline action
                a = rng.choice(legal[i]) if rng.random() < eps else max(legal[i], key=lambda x: Q[s, x])
                r = rew(i, a)
                pb_next = 1 if a == 2 else pb
                if t + 1 < len(sess):
                    j = sess[t + 1]; s2 = S[j][pb_next]
                    if not init[s2]:
                        Q[s2, base_act[j]] = 0.5; init[s2] = True
                    target = r + gamma * max(Q[s2, x] for x in legal[j])
                else:
                    target = r
                N[s, a] += 1; alpha = max(0.03, 1.0 / (1 + N[s, a]) ** 0.7)
                Q[s, a] += alpha * (target - Q[s, a]); tot += r; steps += 1; pb = pb_next
        curve.append(round(tot / max(steps, 1), 4))
        if log_every and (ep + 1) % log_every == 0:
            print(f"  epoch {ep + 1}/{epochs} mean reward {curve[-1]:.3f} eps={eps:.2f}")
    return QPolicy(Q, N, {"lambdas": lam.as_dict(), "epochs": epochs, "gamma": gamma, "visited_states": int(init.sum()),
                          "intent_floor": intent_floor}), curve


def base_risk(rec) -> float:
    from policy.hybrid import risk_score
    return risk_score(rec.an.policy_input(), rec.an.rule_sev)
