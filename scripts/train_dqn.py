"""Pre-train the live deep-RL agent (online DQN) OFFLINE, then it keeps learning online from owner feedback / the labelled simulator.

  python scripts/train_dqn.py                     # ~5-10 min on CPU
  python scripts/train_dqn.py --max-allow 8000 --steps 12000

Training data: TRAIN split only, with out-of-fold classifier/retrieval/supervised features (never in-sample), rewards from the PDF reward
function (policy/q_learning/env.py) evaluated for ALL three actions. Validation (val + val_unseen) is used only to report the result.
Output: models/rl_live/dqn_init.npz and evaluation/results/dqn_pretrain_val.json"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.pipeline import Guardrail
from configs.settings import settings
from evaluation.features import load_rows, oof_analyze, analyze_rows, oof_supervised_actions, rewrite_outcome
from evaluation.metrics import evaluate
from policy.baselines import Hybrid
from policy.hybrid import ACTIONS
from policy.q_learning.dqn import DQNAgent, features, legal_mask
from policy.q_learning.env import reward
from policy.q_learning.live import DQNEvalPolicy, RL_DIR, load_lambdas, rewards_vector
from policy.supervised.model import SupervisedPolicy


def mean_reward(g, recs, acts, lam, floor=0.5):
    tot = 0.0
    for r, a in zip(recs, acts):
        rw = rewrite_outcome(g, r) if a == "REWRITE" else None
        tot += reward(a, r.gt, rw, lam, floor)
    return tot / max(len(recs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-allow", type=int, default=4000); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    g = Guardrail.get(); lam = load_lambdas(); t0 = time.time()
    rows = load_rows("train"); risky = [r for r in rows if r["decision"] != "ALLOW"]; allow = [r for r in rows if r["decision"] == "ALLOW"]
    rnd.shuffle(allow); sample = risky + allow[:a.max_allow]; rnd.shuffle(sample)
    print(f"[1/5] out-of-fold analysis of {len(sample)} train rows ({len(risky)} risky + {min(len(allow), a.max_allow)} benign)", flush=True)
    recs = oof_analyze(sample, g)
    print("[2/5] cross-fitted supervised recommendations", flush=True)
    sup_acts = oof_supervised_actions(recs)
    print("[3/5] building full-information reward targets (all 3 actions per prompt)", flush=True)
    agent = None; kept = 0
    for rec, sa in zip(recs, sup_acts):
        pin = rec.an.policy_input(); pin.sup_action = sa
        if pin.hard_block:
            continue                                   # hard rules are never learned / overridden
        x = features(pin, rec.an.rule_sev)
        if agent is None: agent = DQNAgent(len(x), lr=a.lr, seed=a.seed, capacity=max(len(recs) + 10, 20000))
        rw = rewrite_outcome(g, rec)
        agent.add(x, rewards_vector(rec.gt, rw, lam, 0.5), np.ones(3, np.float32)); kept += 1
    print(f"      kept {kept} transitions; training {a.steps} DQN steps", flush=True)
    agent.batch = 128
    for k in range(a.steps // 500):
        loss = agent.train(500); print(f"  steps {(k + 1) * 500}/{a.steps}  loss {loss:.4f}", flush=True)
    print("[4/5] validation (val + val_unseen) vs the supervised baseline", flush=True)
    sup = SupervisedPolicy.load(); base = Hybrid(None, "hybrid_sup", sup); dqn = DQNEvalPolicy(agent, sup)
    out = {"lambdas": lam.as_dict(), "train_transitions": kept, "steps": a.steps, "sets": {}}
    for sp in ("val", "val_unseen"):
        vr = analyze_rows(load_rows(sp), g)
        if not vr: continue
        res = {}
        for pol in (base, dqn):
            acts = [pol.act(r) for r in vr]
            m = evaluate(g, vr, acts, pol.name); m["mean_reward"] = round(mean_reward(g, vr, acts, lam), 4)
            res[pol.name] = {k: m[k] for k in ("n", "decision_accuracy", "security_recall", "safety_constraint_violation_rate", "false_positive_rate", "legit_blocked_rate", "mean_reward")}
        out["sets"][sp] = res
        print(sp, json.dumps(res, indent=1), flush=True)
    RL_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(RL_DIR / "dqn_init.npz", {"trained_at": time.time(), "lambdas": lam.as_dict(), "transitions": kept, "steps": a.steps, "validation": out["sets"]})
    agent.save(RL_DIR / "live.npz")
    (settings.results_dir / "dqn_pretrain_val.json").write_text(json.dumps(out, indent=2))
    print(f"[5/5] saved models/rl_live/dqn_init.npz in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
