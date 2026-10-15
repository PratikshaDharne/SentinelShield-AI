"""E9 - Live deep-RL (DQN) vs its safe baseline (hard rules + supervised policy): reward, accuracy, safety and paired decision changes.
Rewards use the PDF reward function with the lambdas selected on VALIDATION; the test sets stay isolated."""
from experiments.common import SETS
from evaluation.features import rewrite_outcome
from evaluation.metrics import effective
from policy.q_learning.env import reward
from policy.q_learning.live import load_lambdas


def mean_reward(c, s, acts, lam):
    tot = 0.0
    for rec, a in zip(c.recs[s], acts):
        tot += reward(a, rec.gt, rewrite_outcome(c.g, rec) if a == "REWRITE" else None, lam, 0.5)
    return tot / max(len(acts), 1)


def run(c):
    if "dqn_rl" not in c.policies:
        return {"question": "Does the live deep-RL agent improve the trade-off?", "error": "no trained agent (python scripts/train_dqn.py)"}
    ref, lam = ("hybrid_sup" if "hybrid_sup" in c.policies else "hybrid_base"), load_lambdas()
    res, paired = {}, {}
    for s in SETS:
        b, d = c.run(ref, s), c.run("dqn_rl", s)
        res[s] = {ref: {**c.public(b), "mean_reward": round(mean_reward(c, s, b["_actions"], lam), 4)},
                  "dqn_rl": {**c.public(d), "mean_reward": round(mean_reward(c, s, d["_actions"], lam), 4)}}
        fixed = regressed = changed = 0
        for rec, a0, a1 in zip(c.recs[s], b["_actions"], d["_actions"]):
            if a0 == a1: continue
            changed += 1; ok0, ok1 = effective(c.g, rec, a0)[0] == rec.gt, effective(c.g, rec, a1)[0] == rec.gt
            fixed += (not ok0) and ok1; regressed += ok0 and (not ok1)
        paired[s] = {"changed": changed, "fixed_by_rl": int(fixed), "regressed_by_rl": int(regressed)}
    return {"question": "Does the live deep-RL agent improve the trade-off?", "reference_policy": ref, "lambdas": lam.as_dict(), "results": res, "paired_changes": paired}
