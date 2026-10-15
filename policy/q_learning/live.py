"""Live reinforcement-learning controller (online DQN) for the Allow / Rewrite / Block decision.

How it stays SAFE (PDF section 7 / 20):
  * Hard rules are never routed through RL (base.source == "hard_rule" bypasses this module).
  * The agent may only pick actions inside policy.hybrid.allowed_actions (the safety envelope).
  * It departs from the safe baseline (hard rules + supervised policy) only when Q(agent) - Q(baseline) >= margin.
  * Modes: off | shadow (learn + propose, baseline acts) | active (agent acts) | auto (shadow until a statistical gate says the
    agent beats the baseline on PAIRED live rewards, then active; automatic rollback to shadow if it regresses).
  * Exploration while acting is tiny and limited to the safe side of the baseline (never more permissive).

Where live reward comes from (be honest about it):
  * owner feedback (Review Queue / Playground buttons)  -> the label gives the reward of every action (full information)
  * labelled traffic simulator (real public datasets streamed through the real pipeline, scored by ground truth)
  Without labels there is no true reward - this module never invents one.
"""
from __future__ import annotations
import json, threading, time
from collections import deque
from pathlib import Path
import numpy as np
from configs.settings import settings
from policy.hybrid import ACTIONS, PolicyOutput, allowed_actions
from policy.q_learning.dqn import DQNAgent, features, legal_mask
from policy.q_learning.env import Lambdas, reward

MODES = ("off", "shadow", "active", "auto")
RANK = {"ALLOW": 0, "REWRITE": 1, "BLOCK": 2}
RL_DIR = settings.models_dir / "rl_live"
DEFAULT_RW = {"ok": True, "unsafe": False, "sim_after": 0.5, "ms": 8.0}


def load_lambdas() -> Lambdas:
    f = settings.results_dir / "q_lambda_search.json"
    try:
        return Lambdas(**json.loads(f.read_text())["selected"]["lambdas"])
    except Exception:
        return Lambdas(fn=5.0, fp=1.5, br=1.5, rw=1.0, lat=0.05)


def rewards_vector(gt: str, rw: dict | None, lam: Lambdas, floor: float) -> np.ndarray:
    rw = rw or DEFAULT_RW
    return np.array([reward("ALLOW", gt, None, lam, floor), reward("REWRITE", gt, rw, lam, floor), reward("BLOCK", gt, None, lam, floor)], dtype=np.float32)


def violation(action_idx: int, gt: str, rw: dict | None) -> int:
    a = ACTIONS[action_idx]
    if a == "ALLOW" and gt != "ALLOW": return 1
    if a == "REWRITE" and rw and rw.get("ok") and rw.get("unsafe"): return 1
    return 0


class RLManager:
    def __init__(self, agent: DQNAgent | None = None, lam: Lambdas | None = None, *, mode: str | None = None, margin: float = 0.05,
                 eps_live: float | None = None, window: int = 300, utd: int = 2, intent_floor: float = 0.5, persist: bool = True):
        self.lock = threading.RLock()
        self.agent = agent; self.lam = lam or load_lambdas(); self.floor = intent_floor
        self.margin, self.window, self.utd = margin, window, utd
        self.eps_live = settings.rl_explore_eps if eps_live is None else eps_live
        m = mode or settings.rl_mode
        self.mode = m if (m in MODES and agent is not None) else "off"
        self.acting = False                      # only meaningful in auto mode
        self.hist: deque = deque(maxlen=50000)   # one record per LABELLED event (this is the live reward curve)
        self.gate_log: deque = deque(maxlen=100)
        self.c = {"decisions": 0, "labeled": 0, "hard_rule_bypass": 0, "owner_labels": 0, "sim_labels": 0, "overrides": 0}
        self.rng = np.random.default_rng(7)
        self.listeners: list = []
        self._persist_every, self._since_save, self.persist = 250, 0, persist
        if persist: self._load_history()

    # ------------------------------------------------------------------ construction / persistence
    @classmethod
    def load(cls) -> "RLManager":
        a = DQNAgent.load(RL_DIR / "live.npz") or DQNAgent.load(RL_DIR / "dqn_init.npz")
        return cls(a)

    def _hist_file(self) -> Path: return RL_DIR / "history.jsonl"

    def _load_history(self):
        f = self._hist_file()
        if f.exists():
            try:
                for line in f.read_text().splitlines()[-20000:]:
                    self.hist.append(json.loads(line))
                self.c["labeled"] = len(self.hist)
            except Exception:
                pass

    def save(self):
        if self.agent is None: return
        with self.lock:
            self.agent.save(RL_DIR / "live.npz", {"saved_at": time.time(), "updates": self.agent.updates, "labeled": self.c["labeled"],
                                                  "lambdas": self.lam.as_dict(), "mode": self.mode})

    def set_mode(self, mode: str):
        if mode not in MODES: raise ValueError(mode)
        with self.lock:
            if self.agent is None and mode != "off": raise ValueError("no trained agent: run python scripts/train_dqn.py")
            self.mode = mode; self.acting = False
            self.gate_log.append({"ts": time.time(), "event": f"mode set to {mode}"})

    def reset(self, warm: bool = True, seed: int = 0):
        """warm=True reloads the pretrained agent; warm=False starts from a random network (a live learning demo)."""
        with self.lock:
            base = DQNAgent.load(RL_DIR / "dqn_init.npz")
            if base is None: raise ValueError("no pretrained agent; run python scripts/train_dqn.py")
            if warm: self.agent = base
            else: base.reinit(seed); self.agent = base
            self.hist.clear(); self.gate_log.clear(); self.acting = False
            self.c.update({"labeled": 0, "owner_labels": 0, "sim_labels": 0, "overrides": 0})
            try: self._hist_file().unlink()
            except Exception: pass
            if self.mode == "off": self.mode = "shadow"
            self.gate_log.append({"ts": time.time(), "event": "reset " + ("warm start (pretrained)" if warm else "cold start (random network)")})

    # ------------------------------------------------------------------ the decision
    def is_acting(self) -> bool:
        return self.mode == "active" or (self.mode == "auto" and self.acting)

    def pick(self, p, rule_sev: float, base_idx: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Pure policy: (agent action idx, Q-values, features, legal mask). No side effects."""
        x = features(p, rule_sev); legal = legal_mask(p)
        with self.lock:
            q = self.agent.q(x)
        qm = np.where(legal, q, -1e9); a = int(qm.argmax())
        if legal[base_idx] and q[a] - q[base_idx] < self.margin:
            a = base_idx                      # not clearly better than the safe baseline -> follow the baseline
        return a, q, x, legal

    def decide(self, p, base: PolicyOutput, rule_sev: float):
        """Returns (PolicyOutput override or None, event dict or None)."""
        if self.mode == "off" or self.agent is None: return None, None
        if base.source == "hard_rule":
            self.c["hard_rule_bypass"] += 1; return None, None
        base_idx = ACTIONS.index(base.action)
        agent_idx, q, x, legal = self.pick(p, rule_sev, base_idx)
        acting = self.is_acting(); exec_idx = agent_idx; explored = False
        if acting and self.eps_live > 0 and self.rng.random() < self.eps_live and 0.15 <= base.risk <= 0.95:
            safe = [i for i in range(3) if legal[i] and RANK[ACTIONS[i]] >= RANK[ACTIONS[base_idx]] and i != exec_idx]
            if safe: exec_idx = int(self.rng.choice(safe)); explored = True
        if not acting: exec_idx = base_idx
        self.c["decisions"] += 1
        ev = {"x": x, "legal": legal.astype(np.float32), "q": [round(float(v), 4) for v in q], "base": base_idx, "agent": agent_idx, "exec": exec_idx,
              "acting": acting, "explored": explored, "mode": self.mode, "risk": float(base.risk), "ts": time.time(), "eff": exec_idx, "rw": None}
        if exec_idx != base_idx:
            self.c["overrides"] += 1
            return PolicyOutput(ACTIONS[exec_idx], [f"LEARNED_RL_{ACTIONS[exec_idx]}"] + base.reason_codes, base.risk, base.confidence, True, "rl_dqn"), ev
        return None, ev

    @staticmethod
    def finalize(ev: dict | None, final_action: str, rw: dict | None):
        """Called by the pipeline after the rewrite agent ran: what really happened to the request."""
        if ev is None: return
        ev["eff"] = ACTIONS.index(final_action)
        if rw is not None: ev["rw"] = rw

    def public_event(self, ev: dict | None) -> dict | None:
        if ev is None: return None
        return {"mode": ev["mode"], "acting": ev["acting"], "explored": ev["explored"], "baseline": ACTIONS[ev["base"]], "agent": ACTIONS[ev["agent"]],
                "executed": ACTIONS[ev["exec"]], "q": dict(zip(ACTIONS, ev["q"])), "legal": [a for a, m in zip(ACTIONS, ev["legal"]) if m > 0]}

    # ------------------------------------------------------------------ learning from a label
    def learn(self, ev: dict, gt: str, rw: dict | None = None, *, source: str = "owner", full_info: bool = True, x2=None, legal2=None) -> dict | None:
        """ev: event from decide(); gt: the correct decision (label); rw: rewrite outcome for this prompt if known."""
        if ev is None or self.agent is None: return None
        with self.lock:
            R = rewards_vector(gt, rw if rw is not None else ev.get("rw"), self.lam, self.floor)
            eff = int(ev.get("eff", ev["exec"]))
            M = np.ones(3, np.float32) if full_info else np.eye(3, dtype=np.float32)[int(ev["exec"])]
            self.agent.add(ev["x"], R, M, x2, legal2)
            loss = self.agent.train(self.utd) if self.mode != "off" else None
            rw_eff = ev.get("rw")
            rec = {"t": time.time(), "i": self.c["labeled"], "gt": gt, "src": source, "mode": ev["mode"], "acting": bool(ev["acting"]),
                   "base": ev["base"], "agent": ev["agent"], "exec": ev["exec"], "eff": eff,
                   "r_exec": float(R[eff]), "r_base": float(R[ev["base"]]), "r_agent": float(R[ev["agent"]]),
                   "v_exec": violation(eff, gt, rw_eff), "v_base": violation(ev["base"], gt, None), "v_agent": violation(ev["agent"], gt, None),
                   "loss": None if loss is None else round(float(loss), 5), "updates": self.agent.updates, "explored": bool(ev["explored"])}
            self.hist.append(rec); self.c["labeled"] += 1
            self.c["owner_labels" if source == "owner" else "sim_labels"] += 1
            self._gate()
            if self.persist:
                try:
                    RL_DIR.mkdir(parents=True, exist_ok=True)
                    with self._hist_file().open("a") as fh: fh.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                self._since_save += 1
                if self._since_save >= self._persist_every: self._since_save = 0; self.save()
        for fn in self.listeners:
            try: fn(rec)
            except Exception: pass
        return rec

    # ------------------------------------------------------------------ promotion gate (auto mode)
    def gate_stats(self, n: int | None = None) -> dict:
        h = list(self.hist)[-(n or self.window):]
        if not h: return {"n": 0, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "viol_agent": 0, "viol_base": 0}
        d = np.array([r["r_agent"] - r["r_base"] for r in h]); m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        return {"n": len(h), "mean_diff": round(m, 4), "ci_low": round(m - 1.64 * se, 4), "ci_high": round(m + 1.64 * se, 4),
                "viol_agent": int(sum(r["v_agent"] for r in h)), "viol_base": int(sum(r["v_base"] for r in h))}

    def _gate(self):
        if self.mode != "auto": return
        if not self.acting:
            s = self.gate_stats(self.window)
            if s["n"] >= 200 and s["ci_low"] > 0.0 and s["mean_diff"] >= 0.01 and s["viol_agent"] <= s["viol_base"]:
                self.acting = True; self.gate_log.append({"ts": time.time(), "event": "PROMOTED to active", **s})
        else:
            s = self.gate_stats(100)
            if s["n"] >= 60 and (s["ci_high"] < -0.01 or (s["viol_agent"] - s["viol_base"]) / s["n"] > 0.01):
                self.acting = False; self.gate_log.append({"ts": time.time(), "event": "ROLLED BACK to shadow", **s})

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict:
        with self.lock:
            a = self.agent
            return {"mode": self.mode, "acting": self.is_acting(), "has_agent": a is not None, "margin": self.margin, "eps_live": self.eps_live,
                    "updates": a.updates if a else 0, "replay": a.n if a else 0, "loss_ema": None if (not a or a.loss_ema is None) else round(a.loss_ema, 5),
                    "gamma": a.gamma if a else None, "lr": a.lr if a else None, "counters": dict(self.c), "lambdas": self.lam.as_dict(),
                    "gate": self.gate_stats(self.window), "gate_window": self.window, "gate_log": list(self.gate_log)[-8:]}

    def curve(self, last: int = 600, smooth: int = 50) -> dict:
        h = list(self.hist)[-last:]
        if not h: return {"points": []}
        def roll(key):
            v = np.array([r[key] for r in h], float); out = np.empty_like(v)
            cs = np.cumsum(np.insert(v, 0, 0.0))
            for i in range(len(v)):
                lo = max(0, i + 1 - smooth); out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
            return out
        re, rb, ra = roll("r_exec"), roll("r_base"), roll("r_agent")
        ve, vb, va = roll("v_exec"), roll("v_base"), roll("v_agent")
        cum_gain = np.cumsum([r["r_agent"] - r["r_base"] for r in h])
        pts = []
        for i, r in enumerate(h):
            pts.append({"i": r["i"], "reward_exec": round(float(re[i]), 4), "reward_base": round(float(rb[i]), 4), "reward_agent": round(float(ra[i]), 4),
                        "viol_exec": round(float(ve[i]), 4), "viol_base": round(float(vb[i]), 4), "viol_agent": round(float(va[i]), 4),
                        "cum_gain": round(float(cum_gain[i]), 3), "loss": r["loss"], "acting": r["acting"], "src": r["src"]})
        mix = {k: [0, 0, 0] for k in ("base", "agent")}
        for r in h[-smooth * 2:]:
            mix["base"][r["base"]] += 1; mix["agent"][r["agent"]] += 1
        return {"points": pts, "action_mix": {k: dict(zip(ACTIONS, v)) for k, v in mix.items()}}


class DQNEvalPolicy:
    """Evaluation wrapper (experiments / validation): baseline (hard rules + supervised policy) refined by the DQN inside the safety envelope."""
    name = "dqn_rl"

    def __init__(self, agent: DQNAgent, sup=None, margin: float = 0.05):
        from policy.hybrid import HybridPolicy
        self.mgr = RLManager(agent, mode="shadow", margin=margin, persist=False); self.base = HybridPolicy(q_policy=None, sup=sup)

    def act(self, rec) -> str:
        pin = rec.an.policy_input(); out = self.base.decide(pin, rec.an.rule_sev)
        if out.source == "hard_rule":
            return out.action
        a, _, _, _ = self.mgr.pick(pin, rec.an.rule_sev, ACTIONS.index(out.action))
        return ACTIONS[a]
