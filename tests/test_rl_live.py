"""Safety + learning properties of the live deep-RL layer."""
import numpy as np
import pytest
from policy.hybrid import ACTIONS, PolicyInput, PolicyOutput, allowed_actions
from policy.q_learning.dqn import DQNAgent, features, legal_mask
from policy.q_learning.env import Lambdas, reward
from policy.q_learning.live import RLManager, rewards_vector, violation


def pin(**kw):
    d = dict(rule_codes=[], hard_block=False, has_attack_clause=False, has_task=True, clf_attack_prob=0.1, clf_category="benign", ret_attack_vote=0.0,
             ret_top_sim=0.0, context_risk=False, obfuscated=False, vec=[0.0] * 22)
    d.update(kw); return PolicyInput(**d)


def mgr(prefer: int | None = None, mode="active", eps=0.0):
    ag = DQNAgent(30, seed=3, capacity=2000)
    if prefer is not None:                              # adversarial network: huge Q for one action regardless of input
        ag.net.W[-1][:] = 0.0; ag.net.b[-1][:] = -50.0; ag.net.b[-1][prefer] = 50.0
    return RLManager(ag, Lambdas(fn=5, fp=1.5, br=1.5, rw=1.0, lat=0.05), mode=mode, eps_live=eps, persist=False)


def test_hard_rule_decisions_bypass_rl():
    m = mgr(prefer=0)
    ov, ev = m.decide(pin(hard_block=True), PolicyOutput("BLOCK", ["HARD_BLOCK"], 1.0, 1.0, False, "hard_rule"), 1.0)
    assert ov is None and ev is None


@pytest.mark.parametrize("prefer", [0, 1, 2])
@pytest.mark.parametrize("kw", [dict(has_attack_clause=True, has_task=False), dict(has_attack_clause=True, has_task=True),
                                dict(clf_attack_prob=0.97), dict(clf_attack_prob=0.5), dict(context_risk=True, clf_attack_prob=0.4), dict()])
def test_adversarial_agent_stays_inside_safety_envelope(prefer, kw):
    m = mgr(prefer=prefer, eps=0.5)
    p = pin(**kw); base = PolicyOutput("BLOCK" if kw.get("has_attack_clause") and not kw.get("has_task", True) else "ALLOW", ["X"], 0.5, 0.9)
    legal = set(allowed_actions(p))
    for _ in range(50):
        ov, ev = m.decide(p, base, 0.0)
        assert ACTIONS[ev["agent"]] in legal or ACTIONS[ev["agent"]] == base.action
        if ov is not None:
            assert ov.action in legal


def test_attack_clause_without_task_can_never_be_allowed_by_agent():
    m = mgr(prefer=0)                                              # network that always wants ALLOW
    p = pin(has_attack_clause=True, has_task=False)
    ov, ev = m.decide(p, PolicyOutput("BLOCK", ["NO_LEGIT_INTENT"], 0.9, 0.95), 0.9)
    assert ACTIONS[ev["exec"]] == "BLOCK"


def test_exploration_is_never_more_permissive_than_baseline():
    m = mgr(prefer=0, eps=1.0)
    p = pin(clf_attack_prob=0.45)
    base = PolicyOutput("REWRITE", ["X"], 0.5, 0.9)
    for _ in range(200):
        _, ev = m.decide(p, base, 0.5)
        assert ev["exec"] >= 1                                     # REWRITE or BLOCK, never ALLOW


def test_shadow_mode_never_changes_the_decision():
    m = mgr(prefer=0, mode="shadow", eps=1.0)
    ov, ev = m.decide(pin(clf_attack_prob=0.5), PolicyOutput("REWRITE", ["X"], 0.5, 0.9), 0.5)
    assert ov is None and ev["exec"] == ev["base"]


def test_reward_vector_matches_pdf_reward_function():
    lam = Lambdas(fn=5, fp=1.5, br=1.5, rw=1.0, lat=0.05)
    R = rewards_vector("ALLOW", None, lam, 0.5)
    assert R[0] == pytest.approx(1.0) and R[2] == pytest.approx(-lam.fp)
    R = rewards_vector("BLOCK", None, lam, 0.5)
    assert R[0] == pytest.approx(-lam.fn) and R[2] == pytest.approx(1.0)
    assert violation(0, "BLOCK", None) == 1 and violation(2, "BLOCK", None) == 0


def test_online_learning_improves_reward_and_fills_curve():
    m = mgr(mode="shadow"); rng = np.random.default_rng(0)
    first, last = [], []
    for k in range(1500):
        att = rng.random() < 0.4
        p = pin(clf_attack_prob=0.92 if att else 0.05, ret_attack_vote=0.9 if att else 0.0, vec=[0.0] * 4 + [0.92 if att else 0.05] + [0.0] * 17)
        base = PolicyOutput("ALLOW", ["B"], 0.3, 0.9)
        _, ev = m.decide(p, base, 0.0)
        rec = m.learn(ev, "BLOCK" if att else "ALLOW", source="sim")
        (first if k < 200 else last if k >= 1300 else []).append(rec["r_agent"])
    assert np.mean(last) > np.mean(first) + 0.5
    pts = m.curve(last=1500)["points"]
    assert len(pts) == 1500 and np.mean([q["reward_agent"] for q in pts[-100:]]) > np.mean([q["reward_agent"] for q in pts[:100]])


def test_gate_promotes_only_with_evidence_and_rolls_back():
    m = mgr(mode="auto")
    base = dict(t=0, i=0, gt="ALLOW", src="sim", mode="auto", acting=False, base=0, agent=0, exec=0, eff=0, r_exec=1.0, v_exec=0, v_base=0, v_agent=0, loss=None, updates=0, explored=False)
    for k in range(250):                                           # agent identical to baseline -> no promotion
        m.hist.append({**base, "r_base": 1.0, "r_agent": 1.0}); m._gate()
    assert not m.acting
    for k in range(250):                                           # agent clearly better -> promotion
        m.hist.append({**base, "r_base": 0.0, "r_agent": 1.0, "v_base": 1, "v_agent": 0}); m._gate()
    assert m.acting
    for k in range(120):                                           # agent clearly worse -> rollback
        m.hist.append({**base, "r_base": 1.0, "r_agent": -3.0, "v_base": 0, "v_agent": 1}); m._gate()
    assert not m.acting


def test_off_mode_is_inert():
    m = mgr(mode="off")
    assert m.decide(pin(), PolicyOutput("ALLOW", ["B"], 0.1, 0.9), 0.0) == (None, None)
